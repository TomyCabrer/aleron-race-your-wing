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
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from math import atan, atan2, cos, degrees, hypot, radians, sin, sqrt

import numpy as np

import cars
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
#: The seed lap (K): the schema drive.ml.clone reads. `SEED_LAP_COLS` must
#: match `drive/ml/clone.py`'s SEED_COLS by name; the self-check asserts it.
SEED_LAP_KIND = "carsim-seed-lap-1"
SEED_LAP_HZ = 100
SEED_LAP_DIR = os.path.join("runs", "swarm")
SEED_LAP_COLS = ("t", "x", "y", "psi", "u", "v", "r", "beta", "ay", "util_f",
                 "util_r", "wing_deploy", "delta", "throttle", "brake", "wing_on",
                 "wing_deploy_l", "wing_deploy_r", "top_deploy")

#: The pause menu's DEPLOY SWARM page: its rows, their choices and defaults.
#: `seed`: 'none' = the anchor driver, 'latest' = the newest seed lap the
#: user recorded with K, 'best' = the newest swarm checkpoint saved. `gens`
#: 0 = keep breeding until ESC.
#: `view`: 'replay' draws every scored generation as ghosts and waits for
#: the playback before taking the next; 'fast' takes each generation the
#: moment it is scored (the pool is the only clock, and no trace crosses
#: the pipe). `save`: what ESC in the swarm window does with the best.
#: `car`: what the swarm breeds in -- 'same' = the car you are driving
#: (ballast, garage build and all), else a STOCK library car on your
#: session's settings with its own grip scale: the RACE page's car rule
#: (`_swarm_car`, `Sim._race_car`), so a bot bred in the MX-5 is raced and
#: tested in that same MX-5.
SWARM_MENU_DEFAULTS = dict(pop=24, seed="none", gens=0, T=70.0, view="replay", save="ask",
                           car="same")
SWARM_MENU_CHOICES = dict(pop=(8, 12, 16, 24, 32, 48, 64),
                          seed=("none", "latest", "best"),
                          gens=(0, 3, 5, 10, 20, 50),
                          T=(40.0, 55.0, 70.0, 90.0, 120.0),
                          view=("replay", "fast"),
                          save=("ask", "always", "never"),
                          car=("same",) + tuple(cars.CAR_ORDER))
SWARM_SEED_LABELS = {"none": "None (the built-in driver)",
                     "latest": "Your last seed lap",
                     "best": "Best saved swarm"}
SWARM_VIEW_LABELS = {"replay": "watch every generation",
                     "fast": "off - breed flat out (V)"}
SWARM_SAVE_LABELS = {"ask": "ask on exit",
                     "always": "always on exit",
                     "never": "never (K still saves)"}
SWARM_HELP = [("DEPLOY SWARM", [
    ("Car", "what they breed in: your car, or a stock one"),
    ("", "(then raced and tested in that same car)"),
    ("Cars", "how many cars in each generation"),
    ("Seed", "what generation 0 is bred from: nothing,"),
    ("", "your last seed lap, or the last swarm saved"),
    ("Generations", "0 = keep going until ESC"),
    ("Sim time", "seconds each car gets per generation"),
    ("Replay", "watch each generation as ghosts, or off:"),
    ("", "no cars drawn, the next one starts the moment"),
    ("", "one is scored -- many times faster"),
    ("Save best", "what ESC in the swarm does with the best car;"),
    ("", "ask: ENTER saves it, ESC discards it"),
    ("Seed lap", "puts you on the line, recording; at the line"),
    ("", "again you name the lap and it is the seed"),
    ("Deploy", "runs it in this window; ESC there comes back"),
])]
#: The pause menu's RACE VS BOT page. `bot`: 'none' | 'anchor' (the hand-
#: written driver, `Policy()` with theta = 0) | a checkpoint path under
#: drive/ml/checkpoints. The rival is a SECOND Vehicle -- the session's own
#: car and config -- stepped in lockstep with the user's at the same dt,
#: driven by the policy exactly as `--ml-drive` drives one, drawn as a ghost
#: (no collision: it is a pace car, not a wall) with its own lap timer.
#: The page has RACE_GRID_MAX slots: 'bot' / 'car' for the first, 'bot2' /
#: 'car2' for the second, ... (`race_slot_keys`). A slot opens once the one
#: above it is filled.
RACE_GRID_MAX = 3
RACE_MENU_DEFAULTS = dict(bot="anchor", car="same",
                          bot2="none", car2="same",
                          bot3="none", car3="same")
RACE_BOT_ANCHOR = "anchor"
RACE_CHECKPOINT_DIR = os.path.join("drive", "ml", "checkpoints")
#: What a bot drives: 'same' = the session's car (ballast, wings and all),
#: else a STOCK car from `cars.CARS` on the session's config with that car's
#: own grip scale (`Sim._race_car`). The policy is a trim in the car's own
#: actuator units (`policy.Policy.action`), so a checkpoint bred on one car
#: can be put in another and raced -- which is how a bot is tried in a
#: different car.
RACE_BOT_CARS = ("same",) + tuple(cars.CAR_ORDER)
RACE_START_OFFSET_M = 2.2       # the bot lines up this far LEFT of the user
#: The grid, (n, s) per slot: bot 1 on the user's left, bot 2 on the right,
#: bot 3 a row back on the left. A back-row car's `progress` starts at its
#: s (negative), so every gap is read from the LINE.
RACE_GRID = ((RACE_START_OFFSET_M, 0.0), (-RACE_START_OFFSET_M, 0.0),
             (RACE_START_OFFSET_M, -7.0))
#: A bot's lap timer arms once it has travelled this far from its grid
#: slot. The user's standing start at s = 0 is not a line crossing (the first
#: timed lap is the flying one after it), and a bot on the right of the
#: line or a row back would otherwise log a 'start' in its first metres and
#: time a standing-start lap the user's car never times.
RACE_TIMER_ARM_M = 30.0
RACE_RESPAWN_S = 2.5            # off the map / spun for this long -> back to the line
RACE_RESPAWN_V = 8.0            # m/s it rejoins at
RACE_GAP_HZ = 20                # the (progress, t) trail the time gap is read from
RACE_GAP_KEEP_S = 300.0         # how much of it is kept
C_RIVAL = (255, 140, 43)        # the ghost's colour: the HUD's accent orange
C_RIVALS = (C_RIVAL, (110, 200, 255), (215, 120, 255))   # bot 1, 2, 3
RACE_HELP = [("RACE VS BOT", [
    ("Bot 1..3", "who drives each other car: the built-in driver, or a"),
    ("", "checkpoint the swarm saved (drive/ml/checkpoints);"),
    ("", "a slot opens once the one above it is filled"),
    ("car", "what that bot drives: your car, or a stock one"),
    ("Start", "every car to the line: bot 1 on your left, 2 on"),
    ("", "your right, 3 a row back"),
    ("Stop", "takes the bots off the track"),
    ("Test", "bot 1's lap time in EVERY car, at 1 ms, in the"),
    ("", "background: no window, keep driving meanwhile"),
    ("Delete", "removes bot 1's checkpoint file; select twice"),
    ("Gap", "HUD: + you are behind, - you are ahead, in seconds"),
    ("", "along the track, and in metres, per bot"),
    ("R", "any reset restarts the race from the line"),
])]
RACE_NOTE = ("A bot is a second car with the ML driver at the wheel -- your car, or "
             "any car in the library, so one checkpoint can be tried in three "
             "machines. Bots are ghosts -- you drive through them -- so the race is "
             "against their laps, not their bumpers. A bot restarts from the line "
             "if it leaves the map or spins.")
SWARM_NOTE = ("The swarm is a genetic algorithm over the ML driver: every generation "
              "the best cars are kept and the rest are bred from them, in your car "
              "or a stock one (Car). The window "
              "replays each generation as ghost cars while the next is computed -- "
              "or, with Replay off, breeds flat out and shows the table. "
              "K in the swarm window saves the best as a checkpoint you can race "
              "against or breed from again.")

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
PS_PER_W = 1.0 / 735.5        # metric horsepower, which is what "75 hp" is


def engine_ps(mode: str, car=None) -> int:
    """The Engine setting's power on THIS car, in PS, rounded to 5.

    Reproduces the hard-coded labels above exactly on the Corsa (55 kW ->
    74.8 -> 75; x1.5 -> 112.2 -> 110; x2 -> 149.6 -> 150), which is asserted
    in `self_check`. On the 540i "Sport" is 570 PS, and calling that "150 HP"
    -- which the dicts would -- is the kind of stale readout this whole batch
    is about.
    """
    P = (car if car is not None else CorsaC()).P_max
    ps = P * PS_PER_W * ENGINE_SCALE.get(mode, 1.0)
    return int(round(ps / 5.0) * 5)


def engine_hud(mode: str, car=None) -> str:
    return f"{engine_ps(mode, car)} HP"


def engine_label(mode: str, car=None) -> str:
    """The SETTINGS page row. The Corsa keeps its own wording, because
    "Stock 1.2 16V (75 hp)" says something a number cannot."""
    if car is None or getattr(car, "P_max", 55e3) == 55e3:
        return ENGINE_LABELS[mode]
    stem = {"stock": "Stock", "tuned": "Tuned", "sport": "Sport"}[mode]
    return f"{stem} ({engine_ps(mode, car)} hp)"
ENGINE_DEFAULT = "sport"      # the seat's default: a 75 hp 1.2 is slow from it
# The Car setting: cars.CARS -> the CarSpec the session is built on. 'corsa'
# is the car every scripted number is measured on and the only one this study
# calibrated; the other two are contrasting parameter sets (cars.py's own
# docstring is explicit about what is published and what is `est`).
CAR_MODES = cars.CAR_ORDER
CAR_DEFAULT = cars.CAR_DEFAULT
# The Ballast setting: cars.PointMass -> a new CarSpec. Mass AND station,
# because mass alone is not a weight feature (cars.py). The cycle is a short
# ladder; --ballast takes any value in [0, cars.BALLAST_MAX].
BALLAST_STEPS = cars.BALLAST_KG
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
    car: str = CAR_DEFAULT        # cars.CAR_ORDER -> the CarSpec driven
    ballast: float = 0.0          # kg of added mass, cars.with_masses
    ballast_at: str = cars.BALLAST_DEFAULT   # cars.BALLAST_STATIONS
    engine: str = ENGINE_DEFAULT  # ENGINE_MODES -> VehicleConfig.power_scale
    gearbox: str = "auto"         # GEARBOX_MODES
    abs: bool = True              # VehicleConfig.abs_on
    tc: bool = True               # VehicleConfig.tc_on
    steer_aid: bool = True        # input.KeyboardInput.steer_limit
    wet: str = "patch"            # SURFACE_MODES
    camera: str = "car_up"        # CAMERA_MODES
    sound: str = SOUND_DEFAULT    # SOUND_MODES -> audio.CarSound volume
    path: str = field(default=SETTINGS_PATH, repr=False, compare=False)

    KEYS = ("track", "car", "ballast", "ballast_at", "engine", "gearbox",
            "abs", "tc", "steer_aid", "wet", "camera", "sound")

    def clamp(self) -> "Settings":
        if self.track not in trk.TRACKS:
            self.track = "arena"
        if self.car not in cars.CARS:
            self.car = CAR_DEFAULT
        if self.ballast_at not in cars.BALLAST_LABELS:
            self.ballast_at = cars.BALLAST_DEFAULT
        try:
            self.ballast = min(max(float(self.ballast), 0.0), cars.BALLAST_MAX)
        except (TypeError, ValueError):
            self.ballast = 0.0
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
        """The Engine setting. Nothing else, again.

        It briefly also carried the car's own `engine_scale` (= T_max/110), a
        bodily multiplier on the Corsa's torque curve, because that was the
        only way to give a 440 N.m V8 the right magnitude without touching
        `powertrain.py`. `powertrain.engine_curve(car)` now builds each
        engine's own curve, so that multiplier is retired (and would
        double-count). CONTRACT reconciliation 9 is back to its original
        meaning: `power_scale` is the driver aid and is 1.0 on every scripted
        and headless path."""
        return ENGINE_SCALE.get(self.engine, 1.0)

    @property
    def car_base(self):
        """The fitted car WITHOUT the ballast -- the stations are quoted from
        its axles, so `ballast_point` needs this one."""
        return cars.get(self.car)

    def car_spec(self, extra=()):
        """The `CarSpec` a session is built on: the named car plus the
        ballast plus anything else handed in (the garage charges the three
        fitted wings' mass this way). The stock car with no ballast and no
        wings returns `cars.CORSA_C` itself, unballasted and unperturbed."""
        base = self.car_base
        pts = list(extra)
        if self.ballast > 0.0:
            pts.append(cars.ballast_point(base, self.ballast, self.ballast_at))
        return cars.with_masses(base, pts)

    def ballast_text(self) -> str:
        if self.ballast <= 0.0:
            return "None"
        return (f"{self.ballast:.0f} kg  "
                f"{cars.BALLAST_SHORT[self.ballast_at]}")

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
        if getattr(opts, "car", None):
            self.car = opts.car
        if getattr(opts, "ballast", None) is not None:
            self.ballast = float(opts.ballast)
        if getattr(opts, "ballast_at", None):
            self.ballast_at = opts.ballast_at
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
        opts.car, opts.ballast = self.car, self.ballast
        opts.ballast_at = self.ballast_at
        opts.auto_gearbox = (self.gearbox == "auto")
        return self

    def cycle(self, key: str, d: int = +1) -> None:
        """Step one setting to its next (d=+1, the menu's ENTER / RIGHT) or
        previous (d=-1, LEFT) value. Lists wrap; booleans toggle."""
        d = +1 if d >= 0 else -1

        def step(order, cur):
            return order[(order.index(cur) + d) % len(order)]

        if key == "track":
            self.track = step(trk.TRACK_ORDER, self.track)
        elif key == "car":
            self.car = step(CAR_MODES, self.car)
        elif key == "ballast":
            # snap onto the ladder first: --ballast 137 then ENTER goes to 150
            # (LEFT to 100), not to 137+-25, so the row always shows a value
            # from the list
            if d > 0:
                self.ballast = next((v for v in BALLAST_STEPS
                                     if v > self.ballast + 1e-9), BALLAST_STEPS[0])
            else:
                self.ballast = next((v for v in reversed(BALLAST_STEPS)
                                     if v < self.ballast - 1e-9), BALLAST_STEPS[-1])
        elif key == "ballast_at":
            self.ballast_at = step(cars.BALLAST_STATIONS, self.ballast_at)
        elif key == "engine":
            self.engine = step(ENGINE_MODES, self.engine)
        elif key == "tc":
            self.tc = not self.tc
        elif key == "sound":
            self.sound = step(SOUND_MODES, self.sound)
        elif key == "gearbox":
            self.gearbox = step(GEARBOX_MODES, self.gearbox)
        elif key == "abs":
            self.abs = not self.abs
        elif key == "steer_aid":
            self.steer_aid = not self.steer_aid
        elif key == "wet":
            self.wet = step(SURFACE_MODES, self.wet)
        elif key == "camera":
            self.camera = step(CAMERA_MODES, self.camera)


# The settings whose change is a new session (a new map, or a new CarSpec:
# tyre, wheel stations, static loads, roll block, powertrain). On the page
# these are BROWSED with LEFT / RIGHT and applied with ENTER, so the driver
# can read every option before committing to a rebuild.
RESTART_KEYS = ("track", "wet", "car", "ballast", "ballast_at")


SETTINGS_HELP = [
    ("SETTINGS", [
        ("LEFT / RIGHT", "step the value; map, surface, car and ballast"),
        ("", "only preview here - ENTER / CROSS applies them"),
        ("ENTER / CROSS", "cycle the value (apply a previewed one)"),
        ("ESC / CIRCLE", "back to the pause menu (drops a preview)"),
        ("TAB", "next map (while driving)"),
        ("BACKSPACE", "garage (while driving; touchpad on the pad)"),
    ]),
    ("CAR", [
        ("Corsa C 1.2", "the study's car: every acceptance number is this one"),
        ("MX-5 1.8 / 540i", "contrasting parameter sets - lighter/neutral and"),
        ("", "heavy/powerful. BOTH ARE RWD AND DRIVE THEIR FRONT WHEELS:"),
        ("", "powertrain.py is FWD-only, so their traction is fiction"),
        ("", "and their torque curve is the Corsa's shape, scaled"),
    ]),
    ("BALLAST", [
        ("Mass", "0-200 kg, and it moves everything it really moves:"),
        ("", "axle loads, CG height, CG station (a and b), Izz, Ixx,"),
        ("", "the sprung mass and so the roll - never just m"),
        ("Nose / Seat", "ahead of the front axle, low / at the CG: SEAT is"),
        ("", "the control case and changes mass and nothing else"),
        ("Floor / Boot", "over the rear axle at 0.30 m, which LOWERS the CG /"),
        ("", "on the boot floor at 0.65 m, which RAISES it"),
    ]),
    ("ENGINE", [
        ("Stock", "the 1.2 16V, 75 hp: every scripted number is this car"),
        ("Tuned / Sport", "1.5x / 2x the torque curve, clutch uprated to suit;"),
        ("", "TC keeps the fronts from spinning through 1st"),
        ("", "the car's own engine curve is separate (powertrain.engine_curve)"),
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
SETTINGS_NOTE = ("Map, surface, car and ballast changes restart the session - a "
                 "different CarSpec is a different tyre, roll block and gearbox - "
                 "so LEFT / RIGHT only browse them and ENTER applies; the rest "
                 "apply at once. Saved to runs/settings.json.")


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


def _ml_input(path: str, tr, opts):
    """A trained `drive.ml` policy in the driver's seat, or None.

    Deliberately lazy and deliberately forgiving: `drive/ml` is an OPTIONAL
    sub-package (CONTRACT section 8), so a missing checkpoint or a missing
    numpy must print why and hand the session back to the keyboard rather
    than stop it. Nothing imports `drive.ml` unless --ml-drive is given, which
    is what keeps `drive.ml` additive.

    The policy's `Controls` go through the SAME local `ScriptedInput` the
    acceptance scripts use, so the physics path is untouched.
    """
    try:
        from .ml.env import observe
        from .ml.policy import Policy
        from .ml.baseline import driver_trim
        pol = Policy.load(path)
    except Exception as exc:
        print(f"--ml-drive {path}: {type(exc).__name__}: {exc}")
        print("  falling back to the keyboard")
        return None
    #  minus the ES's learning curve and a swarm's per-generation history
    meta = {k: v for k, v in pol.meta.items() if k not in ("curve", "history")}
    print(f"--ml-drive {path}\n  {meta}")
    if meta.get("track") and meta["track"] != getattr(tr, "name", None):
        print(f"  NOTE: trained on '{meta['track']}', driving '{tr.name}' -- "
              f"it has never seen this track")

    #  The per-car trim the policy was MEASURED with. `env.rollout` passes the
    #  car's lock, wheelbase and planned grip; this path did not, so
    #  `--ml-drive` on an MX-5 drove it with the Corsa's numbers and the window
    #  disagreed with every figure in .handoff/09-ml.md. Cached on the car
    #  object's identity, not recomputed per step: `driver_trim` runs a ramp
    #  steer the first time it sees a car.
    _trim: dict = {}

    def fn(t, veh, track):
        if _trim.get("car") is not veh.car:
            _trim.clear()
            _trim["car"] = veh.car
            _trim["t"] = driver_trim(veh.car, float(veh.cfg.mu_scale))
        tm = _trim["t"]
        return pol.controls(observe(veh, track), Controls,
                            lock_rad=tm["lock_rad"], wheelbase=tm["wheelbase"],
                            ay_plan=tm["ay_plan"], k_us=tm["k_us"])

    return ScriptedInput(fn, vehicle=None, track=tr)


def race_bot_choices() -> list:
    """[(spec, label)] the RACE page cycles through: none, the anchor, then
    YOUR bots -- the swarm's checkpoints, newest first, so RIGHT from the
    anchor is the bot you saved last -- then the trained ones (train.py),
    newest first."""
    import glob
    out = [("none", "None"), (RACE_BOT_ANCHOR, "Built-in driver (the anchor, theta = 0)")]
    paths = sorted(glob.glob(os.path.join(RACE_CHECKPOINT_DIR, "*.json")),
                   key=os.path.getmtime, reverse=True)
    mine = [p for p in paths if os.path.basename(p).startswith("swarm_")]
    for path in mine + [p for p in paths if p not in mine]:
        out.append((path, os.path.basename(path)[:-5]))
    return out


def race_newest_bot():
    """The swarm checkpoint saved last, or None: the RACE page's bot 1 when
    a session starts, so racing your newest bot is ESC > Race > Start."""
    import glob
    c = sorted(glob.glob(os.path.join(RACE_CHECKPOINT_DIR, "swarm_*.json")),
               key=os.path.getmtime)
    return c[-1] if c else None


def race_bot_label(spec) -> str:
    for k, lbl in race_bot_choices():
        if k == spec:
            return lbl
    return os.path.basename(str(spec))[:-5] if str(spec).endswith(".json") else str(spec)


def race_slot_keys(i: int) -> tuple:
    """The RACE page's option keys for grid slot `i` (1-based): ('bot',
    'car'), ('bot2', 'car2'), ..."""
    sfx = "" if i == 1 else str(int(i))
    return "bot" + sfx, "car" + sfx


def race_car_label(name) -> str:
    return "same as mine" if name in ("same", None, "") else f"{cars.car_name(name)} ({name})"


def swarm_car_label(name, session_car) -> str:
    """The Deploy-swarm page's Car row."""
    if name in ("same", None, ""):
        return f"same as mine ({cars.car_name(session_car)})"
    return f"stock {cars.car_name(name)} ({name})"


def _swarm_car(name, car, cfg_kwargs: dict, session_car_name: str) -> tuple:
    """The car a swarm breeds in -> (car, cfg_kwargs, car_name).

    'same' (or None) is the session's own car and config, untouched; a
    `cars.CARS` key is that STOCK car on the session's config (aero,
    assists, power scale) with its own grip scale -- `Sim._race_car`'s
    rule, so the car a bot is bred in is the car the RACE page and its
    Test put it back in. An unknown name breeds in the session's car."""
    if name in (None, "", "same"):
        return car, cfg_kwargs, session_car_name
    if name not in cars.CARS:
        print(f"swarm car {name!r}: not in the library "
              f"({', '.join(cars.CAR_ORDER)}); breeding in your car")
        return car, cfg_kwargs, session_car_name
    c = cars.get(name)
    return c, dict(cfg_kwargs, mu_scale=float(c.mu_scale)), name


def _swarm_state_car(path: str, session_car_name: str) -> str:
    """The car a saved swarm was bred in, for a resume with no car given:
    its `car_name` when that is a library car other than the session's
    (it was bred in a stock car), else 'same'."""
    try:
        with open(path) as f:
            bred = json.load(f).get("car_name")
    except Exception:
        return "same"
    if bred and bred != session_car_name and bred in cars.CARS:
        print(f"swarm: resuming in the car it was bred in, the stock {cars.car_name(bred)}")
        return bred
    return "same"


def race_best_path() -> str:
    """'best' resolved: the newest swarm checkpoint, or the anchor (with a
    note) when none has been saved."""
    import glob
    c = sorted(glob.glob(os.path.join(RACE_CHECKPOINT_DIR, "swarm_*.json")),
               key=os.path.getmtime)
    if not c:
        print("--race best: no swarm checkpoint saved yet; racing the anchor")
        return RACE_BOT_ANCHOR
    return c[-1]


def _load_bot(spec: str, tr):
    """(Policy, label) for a RACE page choice, or None with the reason printed.

    Lazy and forgiving like `_ml_input`: drive/ml is OPTIONAL (CONTRACT
    section 8) and nothing in this module imports it until a bot is asked
    for. 'best' is the newest swarm checkpoint, for the command line."""
    if not spec or spec == "none":
        return None
    if spec == "best":
        spec = race_best_path()
    try:
        from .ml.policy import Policy
        if spec == RACE_BOT_ANCHOR:
            pol = Policy()
        else:
            pol = Policy.load(spec)
    except Exception as exc:
        print(f"race vs bot {spec}: {type(exc).__name__}: {exc}")
        return None
    meta = {k: v for k, v in pol.meta.items() if k not in ("curve", "history", "seed_report")}
    if meta:
        print(f"race vs bot {spec}\n  {meta}")
    if meta.get("track") and meta["track"] != getattr(tr, "name", None):
        print(f"  NOTE: trained on '{meta['track']}', racing on '{tr.name}' -- "
              f"it has never seen this track")
    #  the name over the ghost and on the HUD: short, so three of them fit
    return pol, ("anchor" if spec == RACE_BOT_ANCHOR else race_bot_label(spec))


class Rival:
    """The bot's car: a second `Vehicle` on the same track, the ML policy in
    its seat, stepped once per physics step in lockstep with the user's.

    It never touches the user's Vehicle, the user's LapTimer or the user's
    input, and the user's car never touches it (no collision: the renderer
    draws it as a ghost). Its surfaces are sampled per wheel exactly as
    `Sim._sample_surfaces` does, so it drives the same map. The policy path
    is the `--ml-drive` path (`observe` -> `Policy.controls` with the car's
    own `driver_trim`), so a checkpoint's lap here IS the lap `--ml-drive`
    would set. Everything it needs from `drive.ml` is imported by
    `_load_bot`; this class itself only receives the policy object.
    """

    def __init__(self, policy, car, cfg, track, label: str = "bot",
                 dt: float = DT_PHYS, global_wet: float = 1.0,
                 colour=C_RIVAL, grid=RACE_GRID[0], car_name: str = "same",
                 spec=None):
        from .ml.env import observe, N_OBS
        from .ml.baseline import driver_trim
        self._observe = observe
        self.policy = policy
        self.label = label
        self.colour = tuple(colour)
        self.grid_n, self.grid_s = float(grid[0]), float(grid[1])
        self.car_name = car_name            # 'same' or a cars.CARS key
        self.spec = spec                    # the RACE page choice it was built from
        self.track = track
        self.dt = float(dt)
        self.global_wet = float(global_wet)
        self.veh = Vehicle(car, cfg)
        self.lap = LapTimer(track)
        self._trim = driver_trim(car, float(cfg.mu_scale))
        self._obs = np.empty(N_OBS)
        self.surf_stride = max(1, int(round(1.0 / (SURFACE_LOOKUP_HZ * self.dt))))
        self.gap_stride = max(1, int(round(1.0 / (RACE_GAP_HZ * self.dt))))
        self.mu = [1.0, 1.0, 1.0, 1.0]
        self.crr = [1.0, 1.0, 1.0, 1.0]
        self.on_track4 = [True, True, True, True]
        self.ctl = Controls()
        self.respawns = 0
        self.reset()

    # -- the trail the time gap is read from -----------------------------
    def reset(self, t0: float = 0.0) -> None:
        """Back to its grid slot: `grid_n` left of the centreline, `grid_s`
        along it (a back row is negative -- before the line), standing
        start in 1st like the user. `progress` starts at `grid_s`, so a
        back-row car's gap is read from the LINE like everyone's."""
        tr = self.track
        s0 = self.grid_s
        s0 = (s0 % tr.length) if tr.closed else max(s0, 0.0)
        x, y = trk.point_at(tr, s0, self.grid_n)
        _, _, _, psi_c, _ = trk.project(tr, x, y)
        self.veh.reset(x, y, psi_c, V=0.0, gear=1)
        self.n = 0
        self.t = float(t0)
        self.t0 = float(t0)
        self.s, self.n_lat = s0, self.grid_n
        self._s_prev = None
        self.progress = min(self.grid_s, 0.0)
        self.trail_p: list = []            # progress, m (non-decreasing)
        self.trail_t: list = []            # sim time it was reached
        self.lap.reset()
        self._lost_s = 0.0
        self._sample_surfaces()

    def _sample_surfaces(self) -> None:
        veh = self.veh
        c = veh.car
        cs, sn = cos(veh.psi), sin(veh.psi)
        for i, (bx, by) in enumerate(wheel_positions(c)):
            wx = veh.x + bx * cs - by * sn
            wy = veh.y + bx * sn + by * cs
            mu, crr, on = trk.surface_at(self.track, wx, wy, self.global_wet)
            self.mu[i] = mu
            self.crr[i] = crr
            self.on_track4[i] = on

    def step(self, dt: float, t_now: float) -> None:
        """One physics step. `t_now` is the session clock BEFORE the step."""
        veh, tr = self.veh, self.track
        if self.n % self.surf_stride == 0:
            self._sample_surfaces()
        tm = self._trim
        self._observe(veh, tr, self._obs, mu_here=self.mu[0])
        ctl = self.policy.controls(self._obs, Controls, lock_rad=tm["lock_rad"],
                                   wheelbase=tm["wheelbase"], ay_plan=tm["ay_plan"],
                                   k_us=tm["k_us"])
        self.ctl = ctl
        veh.step(ctl, tuple(self.mu), tuple(self.crr), dt)

        s, n_lat, _k, _p, _i = trk.project(tr, veh.x, veh.y)
        half = 0.5 * tr.width
        active = abs(n_lat) <= half + 2.0
        armed = self.progress - min(self.grid_s, 0.0) > RACE_TIMER_ARM_M
        if self._s_prev is not None:
            ds = s - self._s_prev
            if tr.closed:
                if ds < -0.5 * tr.length:
                    ds += tr.length
                elif ds > 0.5 * tr.length:
                    ds -= tr.length
            if abs(ds) <= 10.0 and active:
                self.progress += ds
            self.lap.update(t_now, self._s_prev, s, dt,
                            all_off_track=not any(self.on_track4),
                            active=active and armed)
        self._s_prev = s
        self.s, self.n_lat = s, n_lat
        self.n += 1
        self.t = t_now + dt

        if self.n % self.gap_stride == 0:
            if not self.trail_p or self.progress >= self.trail_p[-1]:
                self.trail_p.append(self.progress)
                self.trail_t.append(self.t)
                keep = int(RACE_GAP_KEEP_S * RACE_GAP_HZ)
                if len(self.trail_p) > 2 * keep:
                    del self.trail_p[:keep]
                    del self.trail_t[:keep]

        # lost: off the map, or spun to a stop. Back to the last sector
        # line it passed, rolling, rather than a bot parked in the grass.
        lost = (abs(n_lat) > half + 4.0
                or (abs(veh.beta) > 1.05 and hypot(veh.u, veh.v) < 4.0))
        self._lost_s = self._lost_s + dt if lost else 0.0
        if self._lost_s > RACE_RESPAWN_S:
            self._respawn()

    def _respawn(self) -> None:
        tr = self.track
        s0 = 0.0
        if tr.sector_s:
            cands = [v for v in tr.sector_s if v <= self.s]
            s0 = max(cands) if cands else 0.0
        x, y = trk.point_at(tr, s0, self.grid_n)
        _, _, _, psi_c, _ = trk.project(tr, x, y)
        self.veh.reset(x, y, psi_c, V=RACE_RESPAWN_V, gear=2)
        self._s_prev = None
        self._lost_s = 0.0
        self.respawns += 1
        self._sample_surfaces()

    @staticmethod
    def t_at(trail_p, trail_t, p: float):
        """The sim time a trail reached progress `p`, interpolated; None if
        it has not got there yet (or the trail is empty)."""
        if not trail_p or p > trail_p[-1]:
            return None
        import bisect
        i = bisect.bisect_left(trail_p, p)
        if i <= 0:
            return trail_t[0]
        p0, p1 = trail_p[i - 1], trail_p[i]
        f = (p - p0) / (p1 - p0) if p1 > p0 else 1.0
        return trail_t[i - 1] + f * (trail_t[i] - trail_t[i - 1])

    def gap_to(self, user_progress: float, user_trail_p, user_trail_t, t_now: float):
        """(gap_s, gap_m): POSITIVE when the user is behind the bot.

        gap_m is the bot's progress minus the user's. gap_s is how long ago
        the leader passed the point the follower is at now."""
        dm = self.progress - user_progress
        if dm >= 0.0:
            t_ref = self.t_at(self.trail_p, self.trail_t, user_progress)
            gs = (t_now - t_ref) if t_ref is not None else float("nan")
        else:
            t_ref = self.t_at(user_trail_p, user_trail_t, self.progress)
            gs = -(t_now - t_ref) if t_ref is not None else float("nan")
        return gs, dm

    def ghost(self):
        v = self.veh
        return (v.x, v.y, v.psi, self.colour, self.label)


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
        self._pending: dict = {}   # RESTART_KEYS browsed on the page but not
        #                            applied: key -> the value the session runs
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

        # --- the SEED LAP (K): the user decides BEFORE a lap that it is the
        # one the swarm breeds from. Armed -> from the next start-line
        # crossing every 100 Hz row of published state + controls is kept;
        # a complete, VALID lap is written to runs/swarm/seed_*.json and the
        # arm drops. An invalid lap (off track) or a reset discards the rows
        # and keeps the arm. Plain JSON, written here with no import of
        # drive.ml (CONTRACT: drive/ never imports it); drive.ml.clone reads it.
        self.seed_armed = False
        self.seed_rows = None          # None = not recording
        self.seed_stride = max(1, int(round(1.0 / (SEED_LAP_HZ * self.dt))))
        self.seed_saved: str | None = None
        self._seed_msg = ""
        self._seed_msg_until = -1
        self._seed_t0 = 0.0            # sim time the recording opened
        self._seed_valid = True        # no all-wheels-off step since it opened
        #: the finished lap waits here for its name: (rows, lap_s). With a
        #: window the TextPrompt asks; headless it is saved under a stamp.
        self._seed_pending = None
        self._seed_prompt = None
        self._ui_text = None
        self._prompt_was_paused = False
        #: the G key: 0 auto (the car picks the outer flank), +1 left panel,
        #: -1 right panel, 2 BOTH (air brake). Non-zero goes to the physics as
        #: `Controls.wing_cmd` with the top wing left on its own law.
        self.wing_side_mode = 0
        self.pose_prev = (self.veh.x, self.veh.y, self.veh.psi)
        self.events_log: list = []
        self.stop_reason = ""
        self._bound = False

        # pause menu (ESC / OPTIONS): built lazily, render path only
        self.menu = None
        self.has_garage = False            # set by the interactive session
        self.hud_cfg = None                # the garage build's HudData fields
        self._menu_was_paused = False
        self._menu_page = "main"           # 'main' | 'settings' | 'swarm' | 'race'
        self.swarm_opts = dict(SWARM_MENU_DEFAULTS)   # the Deploy-swarm page
        self.swarm_launch = None           # set when the page fires 'Deploy'
        # the RACE VS BOT page. `rival` is the bot's car while a race is on;
        # its trail and the user's own feed the HUD's gap.
        self.race_opts = dict(RACE_MENU_DEFAULTS)
        self._race_delete_armed = None     # checkpoint path awaiting a 2nd 'delete'
        self.rivals: list = []             # the grid: one Rival per filled slot
        self._gap_stride = max(1, int(round(1.0 / (RACE_GAP_HZ * self.dt))))
        self.progress = 0.0                # the user's unwrapped centreline metres
        self.trail_p: list = []
        self.trail_t: list = []
        self._race_msg = ""
        self._race_msg_until = -1
        # the RACE page's Test: bot 1 in every car, headless, in a pool
        self._bot_test = None              # in flight: spec, label, pool, result
        self.bot_test_result = None        # the last one: rows per car
        # lap records (drive/records.py, task 19): the interactive session
        # attaches a LapRecorder; a scripted or headless Sim never has one
        self.recorder = None
        self._rec_msg = ""
        self._rec_msg_until = -1
        self._rec_tag_until = -1           # the lap's place / medal tags: THAT lap's note only
        # the TIME TRIAL / pre-race page (drive/prerace.py, task 20): a
        # prerace.PreRace when this session has one; a PICK on it leaves the
        # build (name, json) here and restarts the session on it
        self.prerace = None
        self.prerace_pick = None
        # the PB and ghost-2 ghosts, the live delta, the sector flash
        # (drive/ghosts.py, task 22): a ghosts.GhostSet in a session with records
        self.ghosts = None
        # the driving tutorial (drive/tutorial.py, task 23): a player session
        # has `progress_file` (runs/progress.json; `progress` above is the
        # race's metres); `tutorial` is the running one, `tutorial_car` = this
        # session drives the tutorial's wing car
        self.progress_file = None
        self.tutorial = None
        self.tutorial_car = False
        self.wing_tutor_start = False      # the Tutorial page's wing-design row (task 24)
        # challenges (drive/challenges.py, task 25): `challenge` is the one
        # being driven (a ChallengeRun: its meter sees every step, event and
        # reset); `challenge_pick` / `challenge_end` ask run_interactive_cli
        # for a session in its class / back out of it; `challenge_build` is
        # (build json, library) for the constraint check, player sessions only
        self.challenge = None
        self.challenge_pick = None
        self.challenge_end = False
        self.challenge_build = None

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
        if self.wing_side_mode and ctl.wing_cmd is None:
            ctl.wing_cmd = {1: (True, False, None), -1: (False, True, None),
                            2: (True, True, None)}[self.wing_side_mode]
        if self.recorder is not None:
            self.recorder.controls(self, ctl)   # quantised + logged (records.py)
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
                if e[0] in ("start", "lap"):
                    self._seed_line(e)
                if self.recorder is not None:
                    self.recorder.event(self, e)
                if self.ghosts is not None:
                    self.ghosts.event(self, e)
                if self.challenge is not None:
                    self.challenge.event(self, e)
        if s_prev is not None and self.rivals:
            ds = (s - s_prev)
            if tr.closed:
                if ds < -0.5 * tr.length:
                    ds += tr.length
                elif ds > 0.5 * tr.length:
                    ds -= tr.length
            if abs(ds) <= 10.0 and abs(n_lat) <= 0.5 * tr.width + 2.0:
                self.progress += ds
            if (self.n + 1) % self._gap_stride == 0 and (
                    not self.trail_p or self.progress >= self.trail_p[-1]):
                self.trail_p.append(self.progress)
                self.trail_t.append(t_prev + dt)
                keep = int(RACE_GAP_KEEP_S * RACE_GAP_HZ)
                if len(self.trail_p) > 2 * keep:
                    del self.trail_p[:keep]
                    del self.trail_t[:keep]
        self._s_prev = s
        for rv in self.rivals:
            # the bots' steps, AFTER the user's and reading nothing of it:
            # the user's Vehicle is bit-identical with or without a rival
            # (asserted by V30 in the self-check)
            rv.step(dt, t_prev)
        if self.seed_rows is not None:
            if not any(self.on_track4):
                self._seed_valid = False
            if self.n % self.seed_stride == 0:
                self._seed_row()

        if self.n % self.skid_stride == 0:
            self._emit_skid()

        self.n += 1
        self.t = self.n * dt               # t = n*dt, NEVER accumulated
        if self.recorder is not None:
            self.recorder.after_step(self)
        if self.challenge is not None:
            self.challenge.step(self)          # the challenge's meter (challenges.py)
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
        if self.recorder is not None:
            self.recorder.discard("reset")     # a teleport is not a lap
        if self.tutorial is not None:
            self.tutorial.command("reset")     # the tutorial's R step and retries
        if self.challenge is not None:
            self.challenge.reset()             # a teleport ends the attempt
        tr = self.track
        s0 = 0.0
        V0 = 0.0
        gear = 1
        if to_checkpoint and tr.sector_s:
            cands = [s for s in tr.sector_s if s <= self.s]
            s0 = max(cands) if cands else 0.0
            V0 = min(hypot(self.veh.u, self.veh.v), 25.0)
            gear = max(self.veh.gear, 1)
        if self.rivals:
            # a race restarts from the line: teleporting one car to a sector
            # line while the others keep lapping is not a gap anyone can read
            to_checkpoint = False
            s0, V0, gear = 0.0, 0.0, 1
            for rv in self.rivals:
                rv.reset()
            self.progress = 0.0
            self.trail_p, self.trail_t = [], []
            self._race_note(f"RACE vs {self._rivals_label()}: GO", 3.0)
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
        if self.seed_rows is not None:
            self.seed_rows = None          # a reset is not a lap; stays armed
            self._seed_note("seed lap: discarded (reset)"
                            + (" - still armed" if self.seed_armed else ""), 3.0)

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
            if self._bot_test is not None:
                self._bot_test_poll()

            if self.paused:
                if self.single_step:
                    self.step_physics(self.dt)
                    self.single_step = False
                self.acc = 0.0
                self.alpha_render = 0.0
                nstep = 0
            else:
                nstep = self.pump(dt_wall)
            if self.tutorial is not None:
                self._tutorial_tick()

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
                if self._seed_prompt is not None and self._seed_prompt.open:
                    self._seed_prompt.draw(self.renderer.screen, self._ui_text)
                self.renderer.present()
                fb = getattr(self.inp, "feedback", None)
                if fb is not None:
                    fb(hud)                      # pad rumble; render loop only
                if self.audio is not None:
                    self.audio.update(hud)       # engine / tyres; render loop only
            del nstep
        self.cancel_bot_test()
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
        # settings.power_scale is now exactly ENGINE_SCALE[mode] again -- the
        # car's own engine reaches from_car through engine_curve(car), not by
        # scaling the Corsa's curve through power_scale.
        v.cfg.power_scale = self.settings.power_scale
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

    def apply_setting(self, key: str, d: int = +1) -> bool:
        """Cycle one setting, apply it live, save. Returns True when the
        session has to be rebuilt (a new map or a new surface set)."""
        s = self.settings
        s.cycle(key, d)
        restart = False
        if self.recorder is not None and key not in ("sound", "camera"):
            self.recorder.discard(f"{key} changed")   # the lap cannot be replayed
            self.recorder.retarget(s, self._pending)   # the engine is in the class,
            #                                             the aids go with the lap
        if key in RESTART_KEYS:
            # A different car, or different ballast, is a different CarSpec:
            # the tyre model, the wheel stations, the static loads, the
            # derived roll block and the powertrain params are all built in
            # Vehicle.__init__, so the session is rebuilt exactly as a new
            # map is. Live-patching veh.car would leave every one of them
            # stale, which is the failure mode this feature exists to avoid.
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
        if restart:
            self._pending.clear()      # this value IS the next session's
        self._save_settings()
        return restart

    def preview_setting(self, key: str, d: int) -> bool:
        """LEFT / RIGHT on the settings page. A live setting is applied as
        ENTER would; a RESTART_KEYS setting only changes the row, and the
        first value it left is remembered so ESC can put it back and a
        save in between does not write the preview. Returns True when the
        row now shows a value the session is not running."""
        s = self.settings
        if key not in RESTART_KEYS:
            self.apply_setting(key, d)
            return False
        orig = self._pending.get(key, getattr(s, key))
        s.cycle(key, d)
        if getattr(s, key) == orig:
            self._pending.pop(key, None)
        else:
            self._pending[key] = orig
        return bool(self._pending)

    def commit_pending(self) -> bool:
        """ENTER on a previewed row: the browsed values become the settings
        (saved), and the session restarts on them. False if none pending."""
        if not self._pending:
            return False
        self._pending.clear()
        self._save_settings()
        return True

    def revert_pending(self) -> None:
        """Leaving the page without ENTER: every browsed row goes back to
        what the session runs."""
        for key, val in self._pending.items():
            setattr(self.settings, key, val)
        self._pending.clear()

    def _save_settings(self) -> None:
        """Save what the session runs, never a preview: the file is what the
        next launch builds."""
        s = self.settings
        if not s.path:
            return
        if not self._pending:
            s.save()
            return
        shown = {k: getattr(s, k) for k in self._pending}
        for k, v in self._pending.items():
            setattr(s, k, v)
        s.save()
        for k, v in shown.items():
            setattr(s, k, v)

    def restart(self) -> None:
        """End this session so the outer loop builds a new one from the
        (already saved) settings, in the same window, with the same car."""
        self.stop_reason = "restart"
        self.quit = True

    # ---------------------------------------------------------------- #
    def handle_event(self, ev: str) -> None:
        """Discrete commands from poll_events(). Unknown strings are ignored."""
        if self._seed_prompt is not None and self._seed_prompt.open:
            if ev == "quit":               # window close; the keys go to the prompt
                self.quit = True
            return
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
            #  the keyboard folds F into its own copy too (input.py), and the
            #  two are OR'd in step_physics: on a car that starts ARMED (a
            #  garage build, --wing) F could never switch the wing off, while
            #  the HUD said OFF. The harness's toggle is the one truth.
            kb = getattr(self.inp, "kb", None)
            if kb is not None and hasattr(kb, "wing_on"):
                kb.wing_on = False
        elif ev == "wing_side":
            self.wing_side_mode = {0: +1, +1: -1, -1: 2, 2: 0}[self.wing_side_mode]
        elif ev == "wet":
            self.global_wet = (MU_WET_SCALE if self.global_wet == 1.0 else 1.0)
            self._sample_surfaces()
            if self.recorder is not None:
                self.recorder.discard("the wet toggle (T)")
            if self.challenge is not None:
                self.challenge.reset()         # a surface change ends the attempt
        elif ev == "marker":
            if self.telem is not None:
                self.telem.mark("marker")
        elif ev == "seed_lap":
            self.seed_armed = not self.seed_armed
            if not self.seed_armed:
                self.seed_rows = None
                self._seed_note("seed lap: disarmed", 3.0)
            else:
                self._seed_note("SEED LAP ARMED - recording starts at the line", 4.0)
        elif ev == "clear_skid":
            self.skid.clear()
        elif ev == "ghosts":
            if self.ghosts is not None:
                self.ghosts.enabled = not self.ghosts.enabled
                self._rec_note("ghosts " + ("on" if self.ghosts.enabled else "off"), 2.0)
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
        c = self.veh.car
        return (f"{self.track.title or self.track.name}   "
                f"{cars.car_name(self.settings.car)} {c.m:.0f} kg "
                f"{100 * c.wdist_f:.0f}% front   lap {self.lap.lap}   "
                f"{wing}   {GEARBOX_HUD.get(self.gearbox, '')}   t {self.t:.1f} s")

    def _menu_show_main(self, idx: int = 0) -> None:
        """The pause page: resume / settings / resets / garage / quit."""
        items = [("Resume", "resume"),
                 ("Settings: map, gearbox, ABS, aids, camera", "settings")]
        if self.prerace is not None:
            items.append(("Time trial: your top 5, medals, the build", "timetrial"))
        if self.progress_file is not None:
            from .tutorial import menu_row
            items.append((menu_row(self.progress_file, self.tutorial), "tutorial"))
        if self.progress_file is not None and self.challenge_build is not None:
            from .challenges import menu_row as ch_row
            items.append((ch_row(self.progress_file, self.challenge), "challenges"))
        items += [("Reset to last sector line", "reset"),
                 ("Full reset (skid marks + timing)", "full_reset")]
        if self.has_garage:
            items.append(("Garage: build the flank panel (3D)", "garage"))
        items.append(("Deploy swarm: learning cars that breed", "swarm"))
        items.append((("Race vs bot: " + self._rivals_label()) if self.rivals
                      else "Race vs bot: the ML driver, in your car or another", "race"))
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
        self.revert_pending()
        self.menu.show(items=items, sections=sections, subtitle=self._menu_subtitle(),
                       note=note, footer=foot, title="PAUSED", idx=idx, columns=2)
        self._menu_page = "main"

    def _settings_items(self) -> list:
        s = self.settings
        p = self._pending
        mark = {k: ("  <- ENTER applies" if k in p else "") for k in RESTART_KEYS}
        rows = [(f"{'Map':<11s}{trk.TRACK_TITLES.get(s.track, s.track)}"
                 f"{mark['track']}", "set:track"),
                (f"{'Car':<11s}{cars.car_name(s.car)}  "
                 f"{s.car_base.m:.0f} kg{mark['car']}", "set:car"),
                (f"{'Ballast':<11s}{s.ballast_text()}{mark['ballast']}", "set:ballast"),
                (f"{'Ballast at':<11s}{cars.BALLAST_LABELS[s.ballast_at]}"
                 f"{mark['ballast_at']}", "set:ballast_at"),
                (f"{'Engine':<11s}"
                 f"{engine_label(s.engine, self.veh.car)}", "set:engine"),
                (f"{'Gearbox':<11s}{GEARBOX_LABELS[s.gearbox]}", "set:gearbox"),
                (f"{'ABS':<11s}{'On' if s.abs else 'Off'}", "set:abs"),
                (f"{'TC':<11s}{'On' if s.tc else 'Off'}", "set:tc"),
                (f"{'Steer aid':<11s}{'On' if s.steer_aid else 'Off'}", "set:steer_aid"),
                (f"{'Surface':<11s}{SURFACE_LABELS[s.wet]}{mark['wet']}", "set:wet"),
                (f"{'Camera':<11s}{CAMERA_LABELS[s.camera]}", "set:camera"),
                (f"{'Sound':<11s}{SOUND_LABELS[s.sound]}", "set:sound")]
        if self.has_garage:
            rows.append(("Garage (3D panel editor)", "garage"))
        rows.append(("Back", "settings_back"))
        return rows

    def _menu_show_settings(self, idx: int = 0) -> None:
        self.menu.show(items=self._settings_items(), sections=SETTINGS_HELP,
                       subtitle=self._menu_subtitle(), note=SETTINGS_NOTE,
                       footer="LEFT / RIGHT browse   ENTER / CROSS cycle, apply   "
                              "ESC / CIRCLE back   BACKSPACE garage",
                       title="SETTINGS", idx=idx,
                       columns=1)
        self._menu_page = "settings"

    def _swarm_seed_available(self, key: str) -> str:
        """What the seed choice would actually use, for the row's label."""
        import glob
        if key == "latest":
            c = sorted(glob.glob(os.path.join(SEED_LAP_DIR, "seed_*.json")),
                       key=os.path.getmtime)
            mine = [p for p in c if f"seed_{self.track.name}_" in os.path.basename(p)]
            return os.path.basename(mine[-1])[5:-5] if mine else "none recorded on this map"
        if key == "best":
            c = sorted(glob.glob(os.path.join("drive", "ml", "checkpoints", "swarm_*.json")),
                       key=os.path.getmtime)
            return os.path.basename(c[-1])[6:-5] if c else "none saved yet"
        return ""

    def _swarm_items(self) -> list:
        o = self.swarm_opts
        avail = self._swarm_seed_available(o["seed"])
        seed = SWARM_SEED_LABELS[o["seed"]] + (f"  [{avail}]" if avail else "")
        armed = ("recording now" if self.seed_rows is not None
                 else "armed (K)" if self.seed_armed else "drive one now")
        cfg = self.veh.cfg
        if getattr(cfg, "has_designed", lambda: False)():
            aero = "garage build"
        elif cfg.wing != "off":
            aero = f"{cfg.wing} flank panel"
        else:                              # the swarm breeds THIS car: no wing,
            aero = "NONE - fit one (garage / Aero)"   # nothing to deploy
        return [(f"{'Car':<13s}{swarm_car_label(o.get('car', 'same'), self.settings.car)}",
                 "set:sw_car"),
                (f"{'Aero':<13s}{aero}", "swarm_aero"),
                (f"{'Cars':<13s}{o['pop']}", "set:sw_pop"),
                (f"{'Seed':<13s}{seed}", "set:sw_seed"),
                (f"{'Generations':<13s}{'until ESC' if not o['gens'] else o['gens']}",
                 "set:sw_gens"),
                (f"{'Sim time':<13s}{o['T']:.0f} s per car", "set:sw_T"),
                (f"{'Replay':<13s}{SWARM_VIEW_LABELS[o.get('view', 'replay')]}", "set:sw_view"),
                (f"{'Save best':<13s}{SWARM_SAVE_LABELS[o.get('save', 'ask')]}", "set:sw_save"),
                (f"{'Seed lap':<13s}{armed}", "swarm_arm"),
                ("Deploy the swarm", "swarm_go"),
                ("Back", "swarm_back")]

    def _menu_show_swarm(self, idx: int = 0) -> None:
        self.menu.show(items=self._swarm_items(), sections=SWARM_HELP,
                       subtitle=self._menu_subtitle(), note=SWARM_NOTE,
                       footer="LEFT / RIGHT change   ENTER / CROSS cycle, select   "
                              "ESC / CIRCLE back",
                       title="DEPLOY SWARM", idx=idx, columns=1)
        self._menu_page = "swarm"

    # ---- the RACE VS BOT page, and the grid it fills --------------------
    @property
    def rival(self):
        """Bot 1 (the first grid slot), or None: the one-bot view the CLI
        and V30 read. The grid itself is `rivals`."""
        return self.rivals[0] if self.rivals else None

    def _race_slots(self) -> list:
        """The filled grid slots, in order: [(i, spec, car_name)]. A slot
        counts only while every slot above it is filled, which is also how
        the page shows them."""
        out = []
        for i in range(1, RACE_GRID_MAX + 1):
            kb, kc = race_slot_keys(i)
            spec = self.race_opts.get(kb, "none")
            if spec in ("none", None, ""):
                break
            out.append((i, spec, self.race_opts.get(kc, "same") or "same"))
        return out

    def _rivals_label(self) -> str:
        return ", ".join(rv.label for rv in self.rivals)

    def _race_items(self) -> list:
        o = self.race_opts
        rows = []
        for i in range(1, RACE_GRID_MAX + 1):
            kb, kc = race_slot_keys(i)
            spec = o.get(kb, "none")
            rows.append((f"{'Bot ' + str(i):<8s}{race_bot_label(spec)}", f"set:race_{kb}"))
            if spec == "none":
                break                      # the next slot opens once this one is filled
            rows.append((f"{'  car':<8s}{race_car_label(o.get(kc, 'same'))}",
                         f"set:race_{kc}"))
        n = len(self._race_slots())
        if self.rivals:
            rows.append((f"Restart the race vs {self._rivals_label()}", "race_go"))
            rows.append(("Stop the race (take the bot%s off)"
                         % ("s" if len(self.rivals) > 1 else ""), "race_stop"))
        elif n == 0:
            rows.append(("Start the race (choose a bot first)", "race_go"))
        elif n == 1:
            rows.append(("Start the race: both cars to the line", "race_go"))
        else:
            rows.append((f"Start the race: you and {n} bots to the line", "race_go"))
        spec = o["bot"]
        if self._bot_test is not None:
            rows.append((f"Testing {self._bot_test['label']} in every car ... "
                         "(select: cancel)", "race_test"))
        elif spec != "none":
            rows.append(("Test bot 1 in every car: lap times, no window", "race_test"))
        if spec not in ("none", RACE_BOT_ANCHOR):
            if self._race_delete_armed == spec:
                rows.append((f"Delete {race_bot_label(spec)}? select again to confirm",
                             "race_delete"))
            else:
                rows.append((f"Delete bot {race_bot_label(spec)} (bot 1's file)", "race_delete"))
        rows.append(("Back", "race_back"))
        return rows

    def delete_bot(self, spec=None) -> bool:
        """Remove a checkpoint file under drive/ml/checkpoints. The built-in
        driver and 'none' cannot be deleted. A race with that bot on the
        grid is stopped first; bot 1's choice steps back to the previous
        entry and any other slot holding it is emptied."""
        spec = self.race_opts["bot"] if spec is None else spec
        if spec in ("none", RACE_BOT_ANCHOR, None):
            self._race_note("race: the built-in driver cannot be deleted", 3.0)
            return False
        ch = [k for k, _ in race_bot_choices()]
        if spec not in ch:
            self._race_note("race: that checkpoint is already gone", 3.0)
            self.race_opts["bot"] = ch[0]
            return False
        if any(rv.spec == spec for rv in self.rivals):
            self.stop_race(quiet=True)
        try:
            os.remove(spec)
        except OSError as exc:
            print(f"delete bot {spec}: {type(exc).__name__}: {exc}")
            self._race_note("race: could not delete (see the terminal)", 4.0)
            return False
        i = ch.index(spec)
        self.race_opts["bot"] = ch[i - 1] if i > 0 else "none"
        for k in range(2, RACE_GRID_MAX + 1):
            kb = race_slot_keys(k)[0]
            if self.race_opts.get(kb) == spec:
                self.race_opts[kb] = "none"
        print(f"deleted bot {spec}")
        self._race_note(f"race: deleted {race_bot_label(spec)}", 3.0)
        return True

    def _menu_show_race(self, idx: int = 0) -> None:
        test = self._bot_test_section()
        self.menu.show(items=self._race_items(),
                       sections=([test] if test else []) + RACE_HELP,
                       subtitle=self._menu_subtitle(), note=RACE_NOTE,
                       footer="LEFT / RIGHT choose a bot and its car   ENTER / CROSS select   "
                              "DELETE twice removes it   ESC / CIRCLE back",
                       title="RACE VS BOT", idx=idx, columns=1)
        self._menu_page = "race"

    def _race_step(self, key: str, d: int) -> None:
        """LEFT / RIGHT on a page row; `key` is 'bot', 'car', 'bot2', ..."""
        ch = ([k for k, _ in race_bot_choices()] if key.startswith("bot")
              else list(RACE_BOT_CARS))
        cur = self.race_opts.get(key)
        i = ch.index(cur) if cur in ch else 0
        self.race_opts[key] = ch[(i + d) % len(ch)]
        self._race_delete_armed = None

    def _race_car(self, car_name):
        """(CarSpec, VehicleConfig) a bot drives. 'same' is the session's own
        car and config, ballast and wings included; a library name is that
        STOCK car on the session's config (aero, assists, power scale) with
        its own grip scale, the way `drive.ml.env.rollout` builds it."""
        if car_name in ("same", None, ""):
            return self.veh.car, self.veh.cfg
        if car_name not in cars.CARS:
            raise ValueError(f"unknown car '{car_name}' (one of {', '.join(cars.CAR_ORDER)})")
        from dataclasses import replace
        car = cars.get(car_name)
        return car, replace(self.veh.cfg, mu_scale=float(car.mu_scale))

    def start_race(self, spec=None) -> bool:
        """Put the grid on the line and restart. `spec` given: that one bot
        alone, in the page's car for slot 1 (V30's path); None: every filled
        slot of the RACE page. False (with a note) when no bot is chosen or
        none can be loaded; a slot that fails to load is skipped with the
        reason printed and the race goes on with the rest."""
        self.stop_race(quiet=True)
        slots = ([(1, spec, self.race_opts.get("car", "same"))] if spec is not None
                 else self._race_slots())
        if not slots:
            self._race_note("race: no bot chosen", 4.0)
            return False
        rivals = []
        for i, sp, car_name in slots:
            loaded = _load_bot(sp, self.track)
            if loaded is None:
                continue
            pol, label = loaded
            try:
                car, cfg = self._race_car(car_name)
                if car_name not in ("same", None, ""):
                    label = f"{label}/{car_name}"
                rivals.append(Rival(pol, car, cfg, self.track, label=label, dt=self.dt,
                                    global_wet=self.global_wet,
                                    colour=C_RIVALS[(i - 1) % len(C_RIVALS)],
                                    grid=RACE_GRID[(i - 1) % len(RACE_GRID)],
                                    car_name=car_name, spec=sp))
            except Exception as exc:
                print(f"race vs bot {sp}: {type(exc).__name__}: {exc}")
        if not rivals:
            self._race_note("race: no bot loaded (see the terminal)", 4.0)
            return False
        self.rivals = rivals
        self.reset(to_checkpoint=False)    # every car to the line
        return True

    def stop_race(self, quiet: bool = False) -> None:
        if self.rivals and not quiet:
            self._race_note("race: the bots are off the track" if len(self.rivals) > 1
                            else "race: the bot is off the track", 3.0)
        self.rivals = []
        self.progress = 0.0
        self.trail_p, self.trail_t = [], []

    def _race_note(self, text: str, secs: float) -> None:
        self._race_msg = text
        self._race_msg_until = self.n + int(secs / self.dt)

    # ---- the RACE page's TEST: one bot in every car, no window ----------
    def start_bot_test(self, spec=None, T=None, workers=None) -> bool:
        """Bot 1 (or `spec`) ALONE in every car the page offers -- yours,
        then each stock car on your settings (`_race_car`, the rule a race
        uses) -- driven headless by `drive.ml.evaluate.bot_lap` at the
        contract's 1 ms, on this session's track and surface, in a pool of
        its own: nothing is drawn and the session keeps running. The
        result lands through `_bot_test_poll`: the page, the HUD and the
        terminal. False (with a note) when there is no bot to test."""
        self.cancel_bot_test()
        spec = self.race_opts.get("bot") if spec is None else spec
        if spec == "best":
            spec = race_best_path()
        if not spec or spec == "none":
            self._race_note("test: choose a bot first", 3.0)
            return False
        try:
            import multiprocessing as mp
            from dataclasses import fields
            from .ml.evaluate import bot_lap, BOT_TEST_T
        except Exception as exc:
            print(f"bot test: drive.ml unavailable ({type(exc).__name__}: {exc})")
            self._race_note("test: drive.ml unavailable (see the terminal)", 4.0)
            return False
        T = float(T or BOT_TEST_T)
        jobs = []
        for name in RACE_BOT_CARS:
            car, cfg = self._race_car(name)
            kw = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}
            #  the rollout has no global wet: the same grip through the config
            kw["mu_scale"] = float(kw["mu_scale"]) * self.global_wet
            jobs.append(dict(spec=spec, car=car, cfg_kwargs=kw, tr=self.track, T=T))
        label = "anchor" if spec == RACE_BOT_ANCHOR else race_bot_label(spec)
        n = len(jobs) if workers is None else max(1, min(int(workers), len(jobs)))
        pool = mp.Pool(n)
        self._bot_test = dict(spec=spec, label=label, T=T, pool=pool,
                              res=pool.map_async(bot_lap, jobs, chunksize=1),
                              t0=time.perf_counter())
        print(f"bot test: {label} alone in "
              + ", ".join(race_car_label(c) for c in RACE_BOT_CARS)
              + f" -- {T:.0f} s each at 1 ms, in the background")
        self._race_note(f"TEST {label}: measuring in {len(jobs)} cars at 1 ms ...", 4.0)
        return True

    def cancel_bot_test(self) -> None:
        bt, self._bot_test = self._bot_test, None
        if bt is not None:
            bt["pool"].terminate()

    @staticmethod
    def _bot_test_cell(r: dict) -> str:
        """One car's result as the page and the terminal show it."""
        if r.get("error"):
            return "error: " + r["error"]
        ended = r.get("ended", "time")
        if r.get("best"):
            return f"{r['best']:.2f} s" + ("" if ended == "time" else f"  (then {ended})")
        if r.get("laps"):
            return f"no flying lap ({r['laps']} lap, {ended} at {r.get('t', 0.0):.0f} s)"
        return f"no lap: {ended} at {r.get('t', 0.0):.0f} s, {r.get('s', 0.0):.0f} m"

    def _bot_test_poll(self) -> None:
        """Collect a finished Test: the page (if open), the HUD, the terminal."""
        bt = self._bot_test
        if bt is None or not bt["res"].ready():
            return
        self._bot_test = None
        try:
            res = bt["res"].get()
        except Exception as exc:
            res = [dict(best=None, error=f"{type(exc).__name__}: {exc}")] * len(RACE_BOT_CARS)
        finally:
            bt["pool"].terminate()
        rows = list(zip(RACE_BOT_CARS, res))
        self.bot_test_result = dict(spec=bt["spec"], label=bt["label"], T=bt["T"],
                                    rows=rows, secs=time.perf_counter() - bt["t0"])
        print(f"bot test: {bt['label']}, best flying lap at 1 ms on "
              f"{self.track.title or self.track.name} ({bt['T']:.0f} s per car, "
              f"{self.bot_test_result['secs']:.1f} s wall)")
        for name, r in rows:
            print(f"  {race_car_label(name):28s} {self._bot_test_cell(r)}")
        short = " | ".join(f"{'mine' if n == 'same' else n} "
                           + (f"{r['best']:.2f}" if r.get("best") else "--")
                           for n, r in rows)
        self._race_note(f"TEST {bt['label']} (1 ms): {short}", 15.0)
        if self.menu is not None and self.menu.open and self._menu_page == "race":
            self._menu_show_race(idx=self.menu.idx)

    def _bot_test_section(self):
        """The page's result block: (title, [(car, lap), ...]) or None."""
        r = self.bot_test_result
        if not r:
            return None
        return (f"TEST: {r['label']}, 1 ms, {self.track.name}",
                [(race_car_label(n), self._bot_test_cell(c)) for n, c in r["rows"]])

    def _race_hud(self) -> str:
        if self.n < self._race_msg_until and self._race_msg:
            return self._race_msg
        if not self.rivals:
            return ""
        parts = []
        for rv in self.rivals:
            gs, dm = rv.gap_to(self.progress, self.trail_p, self.trail_t, self.t)
            gap = f"{gs:+.2f} s" if not math.isnan(gs) else "  --  "
            lap = rv.lap
            last = f"{lap.last_lap:.2f}" if not math.isnan(lap.last_lap) else "--"
            best = f"{lap.best_lap:.2f}" if not math.isnan(lap.best_lap) else "--"
            if len(self.rivals) == 1:
                return (f"BOT {rv.label}  gap {gap} ({dm:+.0f} m)  "
                        f"bot lap {lap.lap} last {last} best {best}")
            parts.append(f"{rv.label} {gap} ({dm:+.0f} m) lap {lap.lap} best {best}")
        return "BOTS  " + "  |  ".join(parts)

    def _hud_msg(self) -> str:
        return " | ".join(m for m in (self._rec_hud(), self._seed_hud(), self._race_hud()) if m)

    # ---- lap records (drive/records.py) ----------------------------------
    def _rec_note(self, text: str, secs: float) -> None:
        self._rec_msg = text
        self._rec_msg_until = self.n + int(round(secs / self.dt))

    def _rec_lap(self, res: dict) -> None:
        """The recorder's callback when a lap closes: the medal it earned
        (drive/medals.py) and the HUD note."""
        from .records import lap_note
        if res.get("valid") and "medal" not in res:
            res["medal"], res["medal_best"] = self._lap_medal(res)
        text, secs = lap_note(res)
        self._rec_note(text, secs)
        self._rec_tag_until = self._rec_msg_until
        print(text)

    def _lap_medal(self, res: dict):
        """(the medal a valid lap earned or None, True when it is the class's
        best yet). The thresholds are drive/medals.py's; the best is kept in
        the class file, written by the recorder's filing thread."""
        try:
            from . import medals
            m = medals.medal_for(res["key"], res["time"])
        except Exception:                  # noqa: BLE001 -- medals are optional
            return None, False
        best = False
        if m and self.recorder is not None:
            book = self.recorder.book
            #  the PB before this lap already earned a medal, stored or not
            #  (a table rebuilt since): that is the one to beat
            m0 = medals.medal_for(res["key"], res.get("pb_before"))
            changed = bool(m0) and book.set_best_medal(res["key"], m0, medals.MEDALS, save=False)
            best = book.set_best_medal(res["key"], m, medals.MEDALS, save=False)
            if best or changed:
                self.recorder.save_later(res["key"])
        return m, best

    def _rec_hud(self) -> str:
        return self._rec_msg if (self._rec_msg and self.n < self._rec_msg_until) else ""

    # ---- the TIME TRIAL / pre-race page (drive/prerace.py) ----------------
    def open_prerace(self) -> bool:
        """The pre-race screen: at a timed session's start, and the pause
        menu's Time trial. False when this session has none."""
        if self.prerace is None or self.renderer is None:
            return False
        if self.menu is None or not self.menu.open:
            self._menu_open()
        self._menu_show_prerace()
        return self._menu_page == "prerace"

    def _prerace_sync(self) -> None:
        """The page shows the class the NEXT lap is filed in (a live engine
        change moves it) and the engine the car has now."""
        pr = self.prerace
        if self.recorder is not None:
            pr.key, pr.book = self.recorder.key, self.recorder.book
        pr.titles["engine"] = engine_label(self.settings.engine, self.veh.car)
        if self.ghosts is not None:
            from .ghosts import slot_label
            pr.ghost_label = slot_label(self.ghosts.slot, pr.book, pr.key)

    def _menu_show_prerace(self, idx: int = 0) -> None:
        try:
            self._prerace_sync()
            pr = self.prerace
            items, secs, sub = pr.items(), pr.sections(), pr.subtitle()
        except Exception as exc:           # noqa: BLE001 -- a bad record file must
            print(f"pre-race screen: {type(exc).__name__}: {exc}")   # not take the
            self.prerace = None            # session down: the pause page instead
            self._menu_show_main()
            return
        self.menu.show(items=items, sections=secs, subtitle=sub,
                       note="", footer="ENTER / CROSS race   ESC / CIRCLE the pause menu   "
                                       "or click a row", title="TIME TRIAL", idx=idx, columns=1)
        self._menu_page = "prerace"

    def _menu_show_prerace_pick(self, idx: int = 0) -> None:
        from .prerace import PICK_HELP
        pr = self.prerace
        self.menu.show(items=pr.pick_items(), sections=PICK_HELP, subtitle=pr.pick_subtitle(),
                       note="", footer="ENTER / CROSS drive it   ESC / CIRCLE back",
                       title="PICK A BUILD", idx=idx, columns=1)
        self._menu_page = "prerace_pick"

    def start_timed(self) -> None:
        """RACE: every car to the line and the clock from the next crossing;
        this build becomes the map's default (`runs/records/last_builds.json`)."""
        pr = self.prerace
        if pr is not None and pr.build_json is not None and not self.tutorial_car:
            #  (the tutorial's plate car is in memory only: never a map's default)
            pr.book.set_last_build(self.track.name, pr.build_name, pr.build_json)
        self.reset(to_checkpoint=False)
        self._rec_note("TIME TRIAL: the clock starts when you cross the line", 4.0)

    def _prerace_event(self, action: str) -> bool:
        """The pre-race and pick pages; False lets the hotkeys (R, SHIFT+R,
        BACKSPACE) fall through to the pause menu's own handling."""
        if action in ("reset", "full_reset", "garage"):
            return False
        pr = self.prerace
        idx = self.menu.idx
        if self._menu_page == "prerace":
            if action == "resume":             # ESC: back to the pause page
                self._menu_show_main()
            elif action == "pr_race":
                self._menu_close()
                self.start_timed()
            elif action == "pr_pick":
                self._menu_show_prerace_pick()
            elif action.endswith("set:pr_ghost") and self.ghosts is not None:
                self.ghosts.step_slot(-1 if action.startswith("prev:") else +1)
                self._menu_show_prerace(idx=idx)
            elif action == "pr_edit" and self.has_garage:   # the garage, on this
                self._menu_close()             # build; its ENTER comes back here
                self.stop_reason = "garage"
                self.quit = True
            else:
                self._menu_show_prerace(idx=idx)
            return True
        if action in ("resume", "pr_back"):
            self._menu_show_prerace(idx=1)
        elif action.startswith("pr_build:"):
            name = action[len("pr_build:"):]
            b = pr.builds.get(name)
            if b is None or (name == pr.build_name and pr.saved()):
                self._menu_show_prerace(idx=1)
            else:                              # a new car: a new session on it
                self.prerace_pick = (name, b)
                self._menu_close()
                self.restart()
        else:
            self._menu_show_prerace_pick(idx=idx)
        return True

    # ---- the driving tutorial (drive/tutorial.py) --------------------------
    def _tutorial_tick(self) -> None:
        """Once per frame after the physics: the tutorial reads the sim and
        says what the session must do (another map, a page, nothing)."""
        tut = self.tutorial
        if tut is None:
            return
        if not tut.active:
            self.tutorial = None
            return
        if self.menu is not None and self.menu.open:
            return                             # a page is up: nothing moves
        act = tut.tick(self)
        if act == "restart" and not self.quit:
            self._rec_note(f"tutorial: to the {tut.map_wanted()}", 3.0)
            self.restart()                     # run_interactive_cli moves the map
        elif act == "page":
            self._menu_show_tutorial_step()

    def _tutorial_ctx(self):
        from types import SimpleNamespace
        return SimpleNamespace(settings=self.settings,
                               key=getattr(self.recorder, "key", None))

    def _menu_show_tutorial_step(self, idx: int = 0) -> None:
        """A page step: paused, Continue / End. No window: it continues."""
        tut = self.tutorial
        if self.renderer is None:
            tut.advance()
            return
        if self.menu is None or not self.menu.open:
            self._menu_open()
        title, sub, note, secs, items = tut.page(self._tutorial_ctx())
        self.menu.show(items=items, sections=secs, subtitle=sub, note=note,
                       footer="ENTER / CROSS continue   ESC / CIRCLE continue   or click a row",
                       title=title, idx=idx, columns=1)
        self._menu_page = "tutorial_step"

    def _menu_show_tutorial(self, idx: int = 0) -> None:
        """The pause menu's Tutorial page: start, continue, skip, end; and
        the garage's wing-design tutorial."""
        from . import tutorial as tu
        self.menu.show(items=tu.menu_items(self.tutorial, self.progress_file,
                                           garage=self.has_garage),
                       sections=tu.step_list(self.tutorial, self.progress_file),
                       subtitle=tu.menu_row(self.progress_file, self.tutorial), note="",
                       footer="ENTER / CROSS select   ESC / CIRCLE back", title="TUTORIAL",
                       idx=idx, columns=1)
        self._menu_page = "tutorial"

    def open_tutorial_offer(self) -> None:
        """The first launch: offer the tutorial (once; runs/progress.json)."""
        from . import tutorial as tu
        if self.renderer is None or self.progress_file is None:
            return
        if self.menu is None or not self.menu.open:
            self._menu_open()
        self.menu.show(items=tu.OFFER_ITEMS, sections=tu.step_list(),
                       subtitle="carsim: a Corsa, a flank wing, a stopwatch", note=tu.OFFER_NOTE,
                       footer="ENTER / CROSS select   ESC / CIRCLE not now   or click a row",
                       title="WELCOME", idx=0, columns=1)
        self._menu_page = "tutorial_offer"

    def _tutorial_begin(self, start=None) -> None:
        from .tutorial import Tutorial
        self.tutorial = Tutorial(self.progress_file, start=start)
        self._menu_close()
        self._rec_note(f"TUTORIAL {self.tutorial.label()}: {self.tutorial.step.title}", 4.0)

    def _tutorial_stop(self, why: str) -> None:
        """The tutorial is over (finished or ended): the session drives on;
        a session on the tutorial's wing car restarts on the player's own."""
        self.tutorial = None
        if self.menu is not None and self.menu.open:
            self._menu_close()
        self._rec_note(why, 4.0)
        if self.tutorial_car:
            self.restart()

    def _tutorial_event(self, action: str) -> bool:
        """The tutorial's pages; False lets the hotkeys (R, SHIFT+R,
        BACKSPACE) fall through to the pause menu's own handling."""
        if action in ("reset", "full_reset", "garage"):
            return False
        page, tut, idx = self._menu_page, self.tutorial, self.menu.idx
        if page == "tutorial_step":            # a page step: ESC continues too
            if tut is None:
                self._menu_close()
            elif action in ("tut_next", "resume"):
                tut.advance()
                self._menu_close()
                if not tut.active:
                    self._tutorial_stop("TUTORIAL DONE - ESC > Time trial for your records")
            elif action == "tut_end":
                tut.end()
                self._tutorial_stop("tutorial ended - ESC > Tutorial continues it")
            else:
                self._menu_show_tutorial_step(idx=idx)
            return True
        if page == "tutorial_offer":
            if action == "tut_start":
                self._tutorial_begin()
            elif action in ("tut_later", "resume"):
                self.progress_file.section("tutorial")["offered"] = True
                self.progress_file.save("tutorial")
                if self.prerace is not None and self.recorder is not None:
                    self._menu_show_prerace()  # what the session would have opened on
                else:
                    self._menu_close()
            else:
                self.open_tutorial_offer()
            return True
        if action in ("resume", "tut_back"):   # the pause menu's Tutorial page
            if action == "tut_back" and tut is not None and tut.active:
                self._menu_close()
            else:
                self._menu_show_main()
                self.menu.idx = [a for _, a in self.menu.items].index("tutorial")
        elif action == "tut_skip" and tut is not None:
            tut.skip()
            self._menu_close()
            if not tut.active:
                self._tutorial_stop("TUTORIAL DONE")
        elif action == "tut_start":
            self._tutorial_begin()
        elif action == "tut_resume":
            from .tutorial import saved_state
            self._tutorial_begin(saved_state(self.progress_file)["step"])
        elif action == "tut_end" and tut is not None:
            tut.end()
            self._tutorial_stop("tutorial ended - ESC > Tutorial continues it")
        elif action == "wt_garage" and self.has_garage:
            self.wing_tutor_start = True       # run_interactive_cli starts it
            self._menu_close()                 # in the garage (drive/wing_tutorial.py)
            self.stop_reason = "garage"
            self.quit = True
        else:
            self._menu_show_tutorial(idx=idx)
        return True

    # ---- challenges (drive/challenges.py) ---------------------------------
    def _challenge_stats(self, ch) -> dict:
        """The current build's stats in the challenge's car."""
        from .challenges import build_stats
        from .records import split_key
        js, lib = self.challenge_build
        return build_stats(js, lib, cars.get(split_key(ch["class"])[1]),
                           self.settings.ballast)

    def _menu_show_challenges(self, idx: int = 0) -> None:
        """The CHALLENGES page: every challenge with its stars and best."""
        from . import challenges as chm
        self._ch_all = chm.load_all()
        self.menu.show(items=chm.list_items(self._ch_all, self.progress_file, self.challenge),
                       sections=chm.LIST_HELP, subtitle=chm.menu_row(self.progress_file,
                                                                     self.challenge),
                       note="", footer="ENTER / CROSS open   ESC / CIRCLE back   or click a row",
                       title="CHALLENGES", idx=idx, columns=1)
        self._menu_page = "challenges"

    def _menu_show_challenge(self, cid: str, idx: int = 0) -> None:
        """One challenge: its class, goal, stars, rules against THIS build,
        your best; Start, or why it cannot."""
        from . import challenges as chm
        ch = self._ch_all[cid]
        try:
            stats = self._challenge_stats(ch)
        except Exception as exc:           # noqa: BLE001 -- a bad build: say so
            stats, why = None, [f"the build cannot be read ({type(exc).__name__})"]
        else:
            why = chm.refusals(ch["constraints"], stats)
        items, secs, note = chm.detail(ch, stats, why, self.progress_file)
        self.menu.show(items=items, sections=secs, subtitle=chm.class_text(ch), note=note,
                       footer="ENTER / CROSS select   ESC / CIRCLE back", title=ch["title"].upper(),
                       idx=idx, columns=1)
        self._menu_page = "challenge"
        self._ch_cur = cid

    def _challenge_event(self, action: str) -> bool:
        if action in ("reset", "full_reset", "garage"):
            return False
        idx = self.menu.idx
        if self._menu_page == "challenges":
            if action in ("resume", "ch_back"):
                self._menu_show_main()
                self.menu.idx = [a for _, a in self.menu.items].index("challenges")
            elif action.startswith("ch:") and action[3:] in getattr(self, "_ch_all", {}):
                self._menu_show_challenge(action[3:])
            elif action == "ch_end" and self.challenge is not None:
                self.challenge_end = True      # run_interactive_cli restores the class
                self._menu_close()
                self.restart()
            else:
                self._menu_show_challenges(idx=idx)
            return True
        if action in ("resume", "ch_list"):
            ids = list(getattr(self, "_ch_all", {}))
            cur = getattr(self, "_ch_cur", None)
            self._menu_show_challenges(idx=ids.index(cur) if cur in ids else 0)
        elif action.startswith("ch_go:"):
            self.challenge_pick = action[len("ch_go:"):]
            self._menu_close()
            self.restart()                     # a session in its class
        else:
            self._menu_show_challenge(self._ch_cur, idx=idx)
        return True

    def _swarm_step(self, key: str, d: int) -> None:
        ch = SWARM_MENU_CHOICES[key]
        cur = self.swarm_opts[key]
        i = ch.index(cur) if cur in ch else 0
        self.swarm_opts[key] = ch[(i + d) % len(ch)]

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
        self.revert_pending()
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
        if self._menu_page in ("prerace", "prerace_pick") and self._prerace_event(action):
            return
        if self._menu_page.startswith("tutorial") and self._tutorial_event(action):
            return
        if self._menu_page in ("challenges", "challenge") and self._challenge_event(action):
            return
        if self._menu_page == "settings":
            idx = self.menu.idx
            if action in ("resume", "settings_back"):
                self._menu_show_main(idx=1)
                return
            if action.startswith(("prev:", "next:")):
                # LEFT / RIGHT: browse the row; nothing restarts here
                if action[5:].startswith("set:"):
                    self.preview_setting(action[9:], -1 if action[0] == "p" else +1)
                    self._menu_show_settings(idx=idx)
                return
            if action.startswith("set:"):
                key = action[4:]
                if key in self._pending:
                    self.commit_pending()  # ENTER on a browsed row: apply it
                    self._menu_close()
                    self.restart()
                elif self.apply_setting(key):
                    self._menu_close()
                    self.restart()
                else:
                    self._menu_show_settings(idx=idx)
                return
        if self._menu_page == "swarm":
            idx = self.menu.idx
            if action in ("resume", "swarm_back"):
                self._menu_show_main()
                self.menu.idx = [a for _, a in self.menu.items].index("swarm")
                return
            if action.startswith(("prev:", "next:")):
                if action[5:].startswith("set:sw_"):
                    self._swarm_step(action[12:], -1 if action[0] == "p" else +1)
                    self._menu_show_swarm(idx=idx)
                return
            if action.startswith("set:sw_"):
                self._swarm_step(action[7:], +1)
                self._menu_show_swarm(idx=idx)
                return
            if action == "swarm_arm":
                # the seed lap, now: back to the line, recording from the
                # standing start; the lap closes at the line and asks its name
                self._menu_close()
                self.seed_armed = False
                self.reset(to_checkpoint=False)
                self._seed_open()
                self._seed_note("SEED LAP: GO - recording from the line; "
                                "cross it again to finish", 5.0)
                return
            if action == "swarm_go":
                self._menu_close()
                self.swarm_launch = dict(self.swarm_opts)
                self.stop_reason = "swarm"
                self.quit = True
                return
            if action in ("reset", "full_reset", "garage"):
                pass                       # the hotkeys fall through below
            else:
                self._menu_show_swarm(idx=idx)
                return
        if self._menu_page == "race":
            idx = self.menu.idx
            if action in ("resume", "race_back"):
                self._race_delete_armed = None
                self._menu_show_main()
                self.menu.idx = [a for _, a in self.menu.items].index("race")
                return
            if action.startswith(("prev:", "next:")):
                if action[5:].startswith("set:race_"):
                    self._race_step(action[14:], -1 if action[0] == "p" else +1)
                    self._menu_show_race(idx=idx)
                return
            if action.startswith("set:race_"):
                self._race_step(action[9:], +1)
                self._menu_show_race(idx=idx)
                return
            if action == "race_go":
                self._menu_close()
                self.start_race()
                return
            if action == "race_stop":
                self._menu_close()
                self.stop_race()
                return
            if action == "race_test":
                if self._bot_test is not None:
                    self.cancel_bot_test()
                    self._race_note("test: cancelled", 3.0)
                else:
                    self.start_bot_test()
                self._menu_show_race(idx=idx)
                return
            if action == "race_delete":
                spec = self.race_opts["bot"]
                if self._race_delete_armed == spec:
                    self._race_delete_armed = None
                    self.delete_bot(spec)
                else:
                    self._race_delete_armed = spec
                self._menu_show_race(idx=idx)
                return
            if action in ("reset", "full_reset", "garage"):
                pass                       # the hotkeys fall through below
            else:
                self._menu_show_race(idx=idx)
                return
        elif action.startswith(("prev:", "next:")):
            return                         # LEFT / RIGHT mean nothing on the pause page
        elif action == "settings":
            self._menu_show_settings()
            return
        elif action == "swarm":
            self._menu_show_swarm()
            return
        elif action == "race":
            self._menu_show_race()
            return
        elif action == "timetrial" and self.prerace is not None:
            self._menu_show_prerace()
            return
        elif action == "tutorial" and self.progress_file is not None:
            self._menu_show_tutorial()
            return
        elif action == "challenges" and self.challenge_build is not None:
            self._menu_show_challenges()
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
    #  THE SEED LAP                                                    #
    # ---------------------------------------------------------------- #
    def _seed_note(self, msg: str, secs: float) -> None:
        self._seed_msg = msg
        self._seed_msg_until = self.n + int(round(secs / self.dt))
        print(msg)

    def _seed_hud(self) -> str:
        if self.n < self._seed_msg_until and self._seed_msg:
            return self._seed_msg
        if self.seed_rows is not None:
            return f"SEED LAP recording  {self.lap.lap_time:6.2f} s"
        if self.seed_armed:
            return "SEED LAP armed"
        return ""

    def _seed_open(self) -> None:
        self.seed_rows = []
        self._seed_t0 = self.t
        self._seed_valid = True
        self._seed_row()

    def _seed_line(self, e) -> None:
        """A start-line crossing: close a recording lap, open a new one.

        A recording opened at the line (K) closes at the next `lap` event; one
        opened at a standing start on the line (the menu's Drive seed lap)
        closes at the next crossing of any kind, which the timer calls
        `start`. Either way the lap is the rows between, and its time the
        crossing minus the opening."""
        if self.seed_rows is not None and e[0] in ("start", "lap"):
            lap_s = float(e[2]) - self._seed_t0
            if lap_s < 5.0:
                return                     # the crossing that opened it
            valid = self._seed_valid and (e[0] != "lap" or self.lap.lap_valid)
            rows = self.seed_rows
            self.seed_rows = None
            if valid and len(rows) > 10:
                self.seed_armed = False
                self._seed_finish(rows, lap_s)
                return
            self._seed_note("seed lap: invalid (off track) - discarded"
                            + (", still armed" if self.seed_armed else ""), 4.0)
        if self.seed_armed and self.seed_rows is None:
            self._seed_open()

    def _seed_row(self) -> None:
        v, c = self.veh, self.ctl
        self.seed_rows.append([
            self.t, v.x, v.y, v.psi, v.u, v.v, v.r, v.beta, v.ay,
            v.util_f, v.util_r, v.wing_deploy,
            c.delta, c.throttle, c.brake, 1.0 if self.wing_on else 0.0,
            v.wing_deploy_l, v.wing_deploy_r, v.top_deploy])

    @staticmethod
    def _seed_safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:40]

    def _seed_finish(self, rows: list, lap_s: float) -> None:
        """The lap is complete and valid: ask its name (window), or save it
        under a time stamp (headless / no keyboard to type on)."""
        kb = getattr(self.inp, "kb", self.inp)
        if self.renderer is None or not hasattr(kb, "key_sink"):
            path = self._seed_save(lap_s, rows)
            self._seed_note(f"SEED LAP SAVED {lap_s:.2f} s -> {path}", 8.0)
            return
        try:
            from . import garage_ui as _gui
            if self._seed_prompt is None:
                self._seed_prompt = _gui.TextPrompt()
                self._ui_text = _gui.Text(_gui.Fonts(1.0))
        except Exception as exc:           # noqa: BLE001
            print(f"seed lap: no name prompt ({type(exc).__name__}: {exc})")
            path = self._seed_save(lap_s, rows)
            self._seed_note(f"SEED LAP SAVED {lap_s:.2f} s -> {path}", 8.0)
            return
        self._seed_pending = (rows, lap_s)
        self._seed_prompt.show(f"SEED LAP {lap_s:.2f} s  -  name it", "",
                               "ENTER save (empty = date stamp)   ESC discard")
        self._prompt_was_paused = self.paused
        self.paused = True
        sm = getattr(self.inp, "set_menu", None)
        if sm is not None:
            sm(True)                       # pedals off while typing
        kb.key_sink = self._seed_prompt_key

    def _seed_prompt_key(self, ev) -> None:
        r = self._seed_prompt.handle(ev)
        if r is None:
            return
        kb = getattr(self.inp, "kb", self.inp)
        kb.key_sink = None
        sm = getattr(self.inp, "set_menu", None)
        if sm is not None:
            sm(False)
        if not self._prompt_was_paused:
            self.unpause()
        rows, lap_s = self._seed_pending
        self._seed_pending = None
        if r != "ok":
            self._seed_note("seed lap: discarded", 3.0)
            return
        path = self._seed_save(lap_s, rows, self._seed_prompt.value)
        self._seed_note(f"SEED LAP SAVED {lap_s:.2f} s -> {path}", 8.0)
        # straight to the swarm page with this lap as the seed: Deploy is one press
        if self.renderer is not None:
            self.swarm_opts["seed"] = "latest"
            self._menu_open()
            self._menu_show_swarm()
            self.menu.idx = len(self.menu.items) - 2

    def _seed_save(self, lap_s: float, rows: list, name: str = "") -> str:
        os.makedirs(SEED_LAP_DIR, exist_ok=True)
        name = self._seed_safe_name(name) or time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SEED_LAP_DIR,
                            f"seed_{self.track.name}_{self.settings.car}_{name}.json")
        d = dict(kind=SEED_LAP_KIND, track=self.track.name, car=self.settings.car,
                 name=name, wing=self.wing, lap_time=lap_s, hz=SEED_LAP_HZ, dt=self.dt,
                 lock_rad=float(getattr(self.veh, "lock_rad", 0.0)),
                 mu_scale=float(self.veh.cfg.mu_scale), global_wet=self.global_wet,
                 settings=self.settings.as_dict(),
                 cols=list(SEED_LAP_COLS), rows=rows)
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            json.dump(d, fh)
        os.replace(tmp, path)
        self.seed_saved = path
        return path

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
            wing_deploy_l=getattr(v, "wing_deploy_l", None),
            wing_deploy_r=getattr(v, "wing_deploy_r", None),
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
            engine=engine_hud(self.settings.engine, v.car),
            eng_load=float(getattr(v, "eng_load", 0.0)),
            track_name=self.track.title or self.track.name,
            car_name=cars.car_name(self.settings.car), mass_kg=v.car.m,
            F_top=float(getattr(v, "F_top", 0.0)), D_top=float(getattr(v, "D_top", 0.0)),
            top_deploy=float(getattr(v, "top_deploy", 0.0)),
            msg=self._hud_msg(),
        )
        if self.recorder is not None:      # the class PB, and where the last lap landed
            d["pb_lap"] = self.recorder.book.pb_time(self.recorder.key)
            last = self.recorder.last
            if last and last.get("pos") and self.n < self._rec_tag_until:
                d["lap_rank"] = f"P{last['pos']}"
            if last and last.get("medal") and self.n < self._rec_tag_until:
                d["lap_medal"] = str(last["medal"])
        if self.rivals:
            d["ghosts"] = [rv.ghost() for rv in self.rivals]
        if self.ghosts is not None:        # the PB and ghost 2, the delta, the flash
            gs = self.ghosts
            gs.sync(self.recorder.key if self.recorder is not None else None)
            g = gs.ghost_tuples(self)
            if g:
                d["ghosts"] = g + d.get("ghosts", [])
            d["delta_s"] = gs.delta(self)
            f = gs.flash_now(self)
            if f:
                d["sector_flash"], d["flash_col"] = f
        if self.tutorial is not None:      # the tutorial's box (drive/tutorial.py)
            d["tutorial"] = self.tutorial.overlay()
        if self.challenge is not None and not d.get("tutorial"):
            d["tutorial"] = self.challenge.overlay(self)   # the same box (challenges.py)
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

    def __init__(self, V_tgt, wing_on=False, gains=(KP_N, KD_PSI, KI_N),
                 modulate=False, ay_plan=AY_MAX_DRY):
        self.V_tgt = V_tgt
        self.wing_on = wing_on
        self.modulate = bool(modulate)
        self.ay_plan = ay_plan         # the grip the driver plans to, m/s^2
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

    def throttle_cap(self, veh, V: float, kt: float) -> float:
        """How much throttle the corner the driver can SEE will take.

        The friction ellipse, but FEEDFORWARD on the path: the lateral demand
        of the corner being driven is `a_y = V^2 * |kappa|`, so a car using
        `u = a_y / ay_plan` of its grip sideways has `sqrt(1 - u^2)` of it
        left for driving.

        Feedforward and not feedback, which took a measurement to settle. The
        first version read the DRIVEN axle's `util_r`, and on the 540i that
        was a step too late every time -- `util_r` only rises once the rear is
        already sliding, by which point a 210 kW car has gone. Reading the
        corner instead means the throttle is already short before the rear
        breaks away. Measured on the arena: `util_r` feedback left the 540i
        1371 m off the track, this leaves it on it.

        Only consulted when `self.modulate` is set, which
        `POWER_GRIP_MODULATE` keeps off for the Corsa, so the acceptance lap
        never reaches this code at all.
        """
        u = (V * V * abs(kt)) / max(self.ay_plan, 1e-6)
        room = sqrt(max(0.0, 1.0 - min(u, 1.0) ** 2))
        return max(room, THR_FLOOR)

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
        if self.modulate:
            thr = min(thr, self.throttle_cap(veh, V, kt))

        self.max_n = max(self.max_n, abs(n))
        self.max_beta = max(self.max_beta, abs(degrees(veh.beta)))
        self.V_hist.append(V)
        self.n_hist.append(n)
        return Controls(delta=delta, throttle=thr, brake=brk,
                        auto_gearbox=True, wing_on=self.wing_on)



# --- generalising the scripted driver to a car that is not the Corsa -------
#  `AY_MAX_DRY`, the steering gains and the lookahead were all calibrated on
#  the Corsa, and on the two library cars the scripted lap left the island:
#  arena max |n| 3.21 m (corsa) against 16.27 m (mx5) and 42.30 m (540i), on a
#  6.0 m half-width. Everything below is expressed as a RATIO against the
#  Corsa's own value, so for the Corsa the ratio is exactly 1.0 and
#  `x * 1.0 == x` -- the acceptance lap is bit-for-bit, not merely close.
def car_ay_peak(car, mu_scale: float = 1.0, roll_dist_f: float = 0.74) -> float:
    """Peak sustainable a_y for THIS car, m/s^2, from `qss.fy_max`/`qss.TYRE`.

    The same statement `qss.residuals` makes, minus the device and the yaw
    split: both axles at capacity, total lateral transfer `m*a_y*h_cg/t`
    divided 0.74/0.26, bisected on `a_y`. It is NOT a replacement for
    `qss.max_ay` (which is the reference truth and is bound to the Corsa at
    module scope) -- it exists only to form a ratio between two cars, and the
    ratio is what is used. Contract sections 4 and 7 both require the tyre
    reference to be `qss.fy_max` with `qss.TYRE`, and this obeys that.

    Pure, deterministic, ~200 cheap iterations. Called once per script setup,
    never from the 1 kHz driver loop.
    """
    W = car.m * G
    Fz_f, Fz_r = W * car.wdist_f, W * (1.0 - car.wdist_f)
    t_bar = car.t
    lo, hi = 0.1, 30.0
    for _ in range(200):
        a = 0.5 * (lo + hi)
        dFz = car.m * a * car.h_cg / t_bar
        cap = (qss.axle_capacity(Fz_f, roll_dist_f * dFz, mu_scale)
               + qss.axle_capacity(Fz_r, (1.0 - roll_dist_f) * dFz, mu_scale))
        lo, hi = (a, hi) if car.m * a < cap else (lo, a)
    return 0.5 * (lo + hi)


_AY_MEAS_CACHE: dict = {}


def car_ay_measured(car, mu_scale: float = 1.0) -> float:
    """Peak a_y from an OPEN-LOOP RAMP STEER of this car, m/s^2, cached.

    `car_ay_peak` above is a closed form and it is good to 0.3 % on the Corsa
    and the MX-5 but **4.1 % HIGH on the 540i** (8.8534 est against 8.5046
    measured) -- it omits the scrub-drag and yaw-balance terms `qss.max_ay`
    carries, and those matter more on a heavy car. 4 % the wrong way means the
    driver plans more grip than the car has, which is exactly how a lap ends
    in the grass, so the planned grip comes from the real thing.

    CONTRACT section 4: quantitative limits come from open-loop ramp steer,
    never from a closed-loop controller. ~3.5 s of simulated time, once per
    (car, mu_scale), at script setup -- never in the driver loop.
    """
    key = (id(car), round(float(mu_scale), 6))
    if key not in _AY_MEAS_CACHE:
        from .vehicle import ramp_steer as _ramp
        _AY_MEAS_CACHE[key] = float(
            _ramp(29.0875, car=car, cfg=VehicleConfig(mu_scale=mu_scale))["peak_ay"])
    return _AY_MEAS_CACHE[key]


#: the Corsa's own value, so every ratio below is exactly 1.0 for the Corsa
AY_PEAK_REF = car_ay_peak(CorsaC())
L_REF = CorsaC().L
#: one Corsa object, so `car_ay_measured`'s cache has a stable key and the
#: reference ramp steer is run once per process, not once per car
_CORSA_REF = CorsaC()
#: power per unit grip, (P_wheel/m)/ay_peak, normalised to the Corsa. Above
#: this the scripted driver MODULATES the throttle on the way out of a corner
#: instead of flooring it. Measured: corsa 1.00, mx5 1.71, 540i 2.16 -- so the
#: trigger is a physical property of the car, and it is exactly inert on the
#: Corsa, whose scripted lap is a frozen acceptance number measured with a
#: driver that floors it. A 210 kW rear-driven saloon with the aids off (which
#: is what reconciliation 9 mandates on every scripted path) simply spins: the
#: 540i left the arena by 31 m and the MX-5 by 390 m with RWD and no
#: modulation. Not flooring it mid-corner is a DRIVER model, like the launch
#: assist and the rev-match blip, not an electronic aid.
POWER_GRIP_MODULATE = 1.25
#: the driver never shuts the throttle completely: below this it is coasting,
#: and a coasting car on the exit of a corner is its own kind of unstable.
THR_FLOOR = 0.15
#: How fast the driver's planned margin fades as the car gets further outside
#: the calibration it was tuned on: `margin * (1 - MARGIN_FADE*(pg - 1))`,
#: where `pg` is `power_grip_ratio`. Exactly `margin` at pg = 1 (the Corsa),
#: so the acceptance lap is bit-for-bit; 0.796 for the 540i at pg 2.16, which
#: is the value a sweep found it needs.
#:
#: A counter-steer term (`delta += k*beta`) was tried FIRST and thrown away:
#: swept over k_beta 0 / 0.5 / 1.0 / 1.5 / 2.5 / 4.0 on the 540i it never
#: completed a lap and made `max |n|` WORSE at the margin that matters
#: (2.74 m at k = 0 against 3.25 at k = 1 and 4.46 at k = 2). The 540i was
#: not losing the lap to a slide it could have caught; it was entering the
#: corner too fast in the first place, and the margin is the honest fix.
MARGIN_FADE = 0.10
MARGIN_MIN = 0.55

def power_grip_ratio(car, mu_scale: float = 1.0) -> float:
    """(P_wheel/m) / peak a_y, as a multiple of the Corsa's. Exactly 1.0 for
    the Corsa, by construction."""
    pg = (car.P_wheel / car.m) / car_ay_peak(car, mu_scale)
    ref = (CorsaC().P_wheel / CorsaC().m) / AY_PEAK_REF
    return pg / ref


def driver_scaling(car, mu_scale: float = 1.0) -> dict:
    """How to retune the scripted driver for `car`. All ratios, all 1.0 for
    the Corsa.

    * `ay` -- the grip the speed profile plans to. Scaled by the car's own
      peak, because planning the Corsa's 8.46 m/s^2 on a 1780 kg saloon that
      can only do 8.36 is what drove it off the track.
    * `kp_n` / `kd_psi` -- the path-following gains, scaled by WHEELBASE.
      `psi_dot = V*delta/L`, so a longer car yaws less per unit steer and
      needs proportionally more of it to hold the same loop gain.
    * `lookahead` -- scaled INVERSELY with grip: less grip is a longer
      braking distance, so the driver has to see the corner sooner. Measured
      100->0 km/h: corsa 49.86 m, mx5 55.35 m, 540i 64.00 m.
    """
    #  the ratio of MEASURED peaks, so it is exactly 1.0 for the Corsa
    #  (the same call on the same object) and honest for everyone else
    ay_ratio = (car_ay_measured(car, mu_scale)
                / car_ay_measured(_CORSA_REF, 1.0))
    L_ratio = car.L / L_REF
    pg = power_grip_ratio(car, mu_scale)
    return dict(ay=AY_MAX_DRY * ay_ratio,
                kp_n=KP_N * L_ratio,
                kd_psi=KD_PSI * L_ratio,
                lookahead=12.0 / ay_ratio,
                ay_ratio=ay_ratio, L_ratio=L_ratio,
                power_grip=pg, modulate=pg > POWER_GRIP_MODULATE,
                margin_scale=min(max(1.0 - MARGIN_FADE * (pg - 1.0),
                                     MARGIN_MIN), 1.0))


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
    #  the car's OWN peak, not the Corsa's constant. Exactly AY_MAX_DRY when
    #  the car IS the Corsa (driver_scaling's ratio is then exactly 1.0).
    ay = margin * driver_scaling(car, getattr(car, "mu_scale", 1.0))["ay"] * mu
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

    def __init__(self, tr, margin=0.90, wing_on=False, global_wet=1.0, car=None):
        #  `car=None` keeps the Corsa construction every acceptance number was
        #  measured on. `speed_profile` was never handed the car, so a 540i lap
        #  planned the Corsa's grip AND capped at the Corsa's Vmax.
        sc = driver_scaling(car or CorsaC(),
                            getattr(car, "mu_scale", 1.0) if car else 1.0)
        super().__init__(0.0, wing_on=wing_on,
                         gains=(sc["kp_n"], sc["kd_psi"], KI_N),
                         modulate=sc["modulate"], ay_plan=sc["ay"])
        #  a car this far outside the driver's calibration gets a more
        #  careful driver, which is what a human does in an unfamiliar
        #  overpowered car. Exactly `margin` for the Corsa.
        self.margin = margin * sc["margin_scale"]
        self.prof = speed_profile(tr, self.margin, car=car,
                                  global_wet=global_wet)
        self.ds = tr.ds
        self.N = len(self.prof)
        self.lookahead = sc["lookahead"]  # m, so the driver brakes BEFORE the corner
        self.scaling = sc

    def target(self, t, veh, tr, s, kt):
        i = int((s + self.lookahead) / self.ds) % self.N
        return float(self.prof[i])


# ==================================================================== #
#  SCRIPTS                                                             #
# ==================================================================== #
def _build(track_name="arena", radius=50.0, cw=False, wing="off",
           x_w=0.97, h_w=0.90, wet="patch", dt=DT_PHYS, telem_path=None,
           telem_hz=TELEM_HZ, precision="6g", driver=None, mu_scale=None,
           cmdline=None, start_V=0.0, gear=1, tag="", start_s=None, car=None,
           dev_flank="outer"):
    """One place that assembles a headless Sim, so every script agrees.

    `car=None` is `CorsaC()` -- not an equal copy, the same construction
    every acceptance number was measured on. `mu_scale=None` is the car's
    own `mu_scale` (1.0 for the Corsa), and an explicit value still wins,
    which is what the wet rigs pass.
    """
    tr = trk.make_track(track_name, radius, cw, surfaces=(wet != "none"))
    global_wet = MU_WET_SCALE if wet == "all" else 1.0

    if mu_scale is None:
        mu_scale = getattr(car, "mu_scale", 1.0) if car is not None else 1.0
    # 1.0, always, on every scripted path (reconciliation 9). The car's own
    # engine now reaches the physics through `powertrain.engine_curve(car)`
    # rather than by scaling the Corsa's curve through here, so the old
    # `p_scale = car.engine_scale` is gone -- it would double-count.
    p_scale = 1.0
    cfg = VehicleConfig(wing=wing, x_w=x_w, h_w=h_w, mu_scale=mu_scale,
                        power_scale=p_scale, dev_flank=dev_flank)
    veh = Vehicle(CorsaC() if car is None else car, cfg)
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
                    wing=dict(wing=wing, x_w=x_w, h_w=h_w, dev_flank=dev_flank),
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


def _opts_car(opts):
    """The `CarSpec` a scripted / headless run drives, or **None** for the
    stock Corsa C with no ballast.

    None, not `cars.CORSA_C`: `_build` then constructs `CorsaC()` exactly as
    it always did, so the default path is identical rather than merely equal
    (the telemetry sidecar serialises this object, among other things).
    """
    name = getattr(opts, "car", None) or CAR_DEFAULT
    kg = float(getattr(opts, "ballast", 0.0) or 0.0)
    if name == CAR_DEFAULT and kg <= 0.0:
        return None
    base = cars.get(name)
    if kg <= 0.0:
        return base
    where = getattr(opts, "ballast_at", cars.BALLAST_DEFAULT)
    return cars.with_masses(base, [cars.ballast_point(base, kg, where)])


# ---- accel ---------------------------------------------------------------
def accel_script(opts) -> dict:
    """WOT from rest. The gearbox shifts itself (powertrain.update_shift)."""
    drv = StraightDriver(throttle=1.0, wing_on=(opts.wing != "off"))
    sim = _build("dragstrip", wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                 wet=opts.wet, dt=opts.dt, telem_path=opts.telemetry,
                 telem_hz=opts.telem_hz, precision=opts.telem_precision,
                 driver=drv, tag="accel", car=_opts_car(opts))
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
                 precision=opts.telem_precision, driver=drv, tag="brake",
                 car=_opts_car(opts))
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
                     precision="6g", mu_scale=None, car=None,
                     dev_flank="outer"):
    """One 8 s constant-speed window. Returns (held, diagnostics)."""
    drv = PathFollower(V_tgt, wing_on=(wing != "off"))
    sim = _build("skidpad", radius=radius, cw=cw, wing=wing, x_w=x_w, h_w=h_w,
                 wet=wet, dt=dt, telem_path=telem_path, telem_hz=telem_hz,
                 precision=precision, driver=drv, mu_scale=mu_scale,
                 start_V=V_tgt, gear=3, tag=f"skid{radius:g}", car=car,
                 dev_flank=dev_flank)
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
    _car = _opts_car(opts)
    lo, hi = 5.0, 1.25 * sqrt(AY_MAX_DRY * R) + 2.0
    ok_lo = False
    diag_lo = {}
    for _ in range(6):
        held, d = _skidpad_attempt(lo, R, wing, opts.wing_x, opts.wing_h,
                                   opts.wet, opts.dt, opts.cw, car=_car,
                                   dev_flank=getattr(opts, "dev_flank", "outer"))
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
                                   opts.wet, opts.dt, opts.cw, car=_car,
                                   dev_flank=getattr(opts, "dev_flank", "outer"))
        if held:
            lo, diag_lo = mid, d
        else:
            hi = mid
    # final holding run, logged
    tp = telem_path if telem_path is not None else opts.telemetry
    held, diag = _skidpad_attempt(lo, R, wing, opts.wing_x, opts.wing_h,
                                  opts.wet, opts.dt, opts.cw,
                                  telem_path=tp, telem_hz=opts.telem_hz,
                                  precision=opts.telem_precision, car=_car,
                                  dev_flank=getattr(opts, "dev_flank", "outer"))
    out = dict(script="skidpad_limit", radius=R, wing=wing, V_limit=lo,
               csv=tp, **diag)
    try:
        from .vehicle import steady_state_corner
        ref = steady_state_corner(R, cfg=VehicleConfig(
            wing=wing, x_w=opts.wing_x, h_w=opts.wing_h,
            dev_flank=getattr(opts, "dev_flank", "outer")))
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
                            mu_scale=mu, dev_flank=getattr(opts, "dev_flank", "outer"))
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
                                   mu_scale=mu, car=_opts_car(opts),
                                   dev_flank=getattr(opts, "dev_flank", "outer"))
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
                 precision=opts.telem_precision, driver=drv, tag="ramp",
                 car=_opts_car(opts))
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
                    global_wet=(MU_WET_SCALE if opts.wet == "all" else 1.0),
                    car=_opts_car(opts))
    sim = _build("arena", wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                 wet=opts.wet, dt=opts.dt, telem_path=opts.telemetry,
                 telem_hz=opts.telem_hz, precision=opts.telem_precision,
                 driver=drv, start_V=25.0, gear=3, tag="lap",
                 car=_opts_car(opts), dev_flank=getattr(opts, "dev_flank", "outer"))
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
                 wing_x=0.97, wing_h=0.90, wing_inc=0.0, dev_flank="outer",
                 dt=DT_PHYS, duration=60.0,
                 telemetry=None, telem_hz=TELEM_HZ, telem_precision="6g",
                 margin=0.90, gearbox="auto", auto_gearbox=True, abs=False,
                 steer_limit=True, camera="car_up", engine="stock", tc=False,
                 sound="off", car=CAR_DEFAULT, ballast=0.0,
                 ballast_at=cars.BALLAST_DEFAULT)
        d.update(kw)
        self.__dict__.update(d)


def _v30_race_vs_bot(verbose=True):
    """The rival is additive: the user's car is bit-identical with and
    without one; the bot itself moves, laps, and comes back to the line
    on a reset; the menu page reaches it without a window. Skipped (as a
    pass) when drive/ml or numpy is unavailable -- the package is optional."""
    from types import SimpleNamespace
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            loaded = _load_bot(RACE_BOT_ANCHOR, trk.make_arena())
    except Exception:
        loaded = None
    if loaded is None:
        if verbose:
            print("  [V30] race vs bot: drive/ml unavailable -- skipped")
        return True, {"skipped": True}

    T = 12.0

    def run(with_bot):
        drv = lambda t, v, T_: Controls(throttle=0.6 if t > 0.5 else 0.0,
                                        delta=radians(2.0) * sin(0.4 * t),
                                        auto_gearbox=True)
        sim = _build("arena", driver=drv)
        if with_bot:
            with contextlib.redirect_stdout(io.StringIO()):
                assert sim.start_race(RACE_BOT_ANCHOR)
        sim.run_headless(T)
        v = sim.veh
        return sim, (v.x, v.y, v.psi, v.u, v.v, v.r, v.rpm, v.gear)

    with contextlib.redirect_stdout(io.StringIO()):
        s0, a = run(False)
        s1, b = run(True)
    same = (a == b)
    rv = s1.rival
    bot_m, user_m = rv.progress, s1.progress
    moved = rv is not None and bot_m > 30.0 and user_m > 30.0
    gs, dm = rv.gap_to(s1.progress, s1.trail_p, s1.trail_t, s1.t)
    gap_ok = math.isfinite(dm) and (math.isnan(gs) or abs(gs) < T)
    hud = s1.hud_data()
    hud_ok = (len(getattr(hud, "ghosts", [])) == 1 and hud.ghosts[0][3] == C_RIVAL
              and "BOT" in hud.msg)
    # a reset puts both on the line and clears the gap
    s1.reset(to_checkpoint=True)
    rst_ok = (rv.progress == 0.0 and s1.progress == 0.0 and s1.s == 0.0
              and abs(hypot(rv.veh.u, rv.veh.v)) < 1e-9)
    # the page, without a window: main -> race -> LEFT/RIGHT cycle -> stop
    with contextlib.redirect_stdout(io.StringIO()):
        s1.renderer = SimpleNamespace(cfg=SimpleNamespace(mode="car_up"))
        s1._menu_open()
        s1.handle_event("nav_down")
        while s1.menu.action() != "race":
            s1.handle_event("nav_down")
        s1.handle_event("select")
        page_ok = s1._menu_page == "race" and s1.menu.action() == "set:race_bot"
        before = s1.race_opts["bot"]
        s1.handle_event("nav_right")
        s1.handle_event("nav_left")
        cyc_ok = s1.race_opts["bot"] == before
        while s1.menu.action() != "race_stop":
            s1.handle_event("nav_down")
        s1.handle_event("select")
        stop_ok = s1.rival is None and not s1.menu.open
        s1.renderer = None
    # a GRID: two bots, the second in a DIFFERENT car (the stock 540i, its own
    # grip scale) on the user's right; both move, both are drawn in their own
    # colour, the HUD reads both gaps, the page shows a car row per filled
    # slot and opens the next slot
    with contextlib.redirect_stdout(io.StringIO()):
        drv2 = lambda t, v, T_: Controls(throttle=0.6 if t > 0.5 else 0.0,
                                         delta=radians(2.0) * sin(0.4 * t),
                                         auto_gearbox=True)
        s2 = _build("arena", driver=drv2)
        s2.race_opts.update(bot=RACE_BOT_ANCHOR, car="same",
                            bot2=RACE_BOT_ANCHOR, car2="540i")
        grid_ok = bool(s2.start_race()) and len(s2.rivals) == 2
        r1, r2 = s2.rivals
        c540 = cars.get("540i")
        car_ok = (r1.veh.car is s2.veh.car and r2.veh.car is c540
                  and abs(r2.veh.cfg.mu_scale - c540.mu_scale) < 1e-12
                  and r2.veh.cfg.wing == s2.veh.cfg.wing
                  and r2.label.endswith("/540i") and r1.label == r2.label[:-5])
        n1 = trk.project(s2.track, r1.veh.x, r1.veh.y)[1]
        n2 = trk.project(s2.track, r2.veh.x, r2.veh.y)[1]
        pos_ok = (abs(n1 - RACE_START_OFFSET_M) < 0.05
                  and abs(n2 + RACE_START_OFFSET_M) < 0.05)
        s2.run_headless(8.0)
        moved2 = all(rv.progress > 20.0 for rv in s2.rivals)
        h2 = s2.hud_data()
        ghosts_ok = (len(h2.ghosts) == 2 and h2.ghosts[0][3] != h2.ghosts[1][3]
                     and h2.ghosts[0][3] == C_RIVAL and "BOTS" in h2.msg
                     and "/540i" in h2.msg)
        s2.renderer = SimpleNamespace(cfg=SimpleNamespace(mode="car_up"))
        s2._menu_open()
        s2._menu_show_race()
        acts = [a for _, a in s2.menu.items]
        rows_ok = (acts[:5] == ["set:race_bot", "set:race_car", "set:race_bot2",
                                "set:race_car2", "set:race_bot3"]
                   and "set:race_car3" not in acts)
        s2._race_step("car", +1)
        cyc2 = s2.race_opts["car"] == RACE_BOT_CARS[1]
        s2.stop_race(quiet=True)
        # the list: none, the anchor, then YOUR (swarm) bots newest first, so
        # RIGHT from the anchor is the bot saved last -- and that is the one
        # a session starts with as bot 1
        ch = [k for k, _ in race_bot_choices()]
        mine = [os.path.basename(k).startswith("swarm_") for k in ch[2:]]
        mt = [os.path.getmtime(k) for k, m in zip(ch[2:], mine) if m]
        order_ok = (ch[:2] == ["none", RACE_BOT_ANCHOR] and mine == sorted(mine, reverse=True)
                    and mt == sorted(mt, reverse=True)
                    and race_newest_bot() == (ch[2] if any(mine) else None))
        cyc2 = cyc2 and order_ok
        # the swarm page's Car row cycles the library, and `_swarm_car` puts
        # a stock car on the session's config the way `_race_car` does
        s2._menu_show_swarm()
        sw_row = "set:sw_car" in [a for _, a in s2.menu.items]
        s2._swarm_step("car", +1)
        kw0 = dict(wing="plate", mu_scale=1.0, abs_on=True)
        c_m, kw_m, n_m = _swarm_car("mx5", s2.veh.car, kw0, "corsa")
        c_s, kw_s, n_s = _swarm_car("same", s2.veh.car, kw0, "corsa")
        swcar_ok = (sw_row and s2.swarm_opts["car"] == SWARM_MENU_CHOICES["car"][1]
                    and c_m is cars.get("mx5") and n_m == "mx5" and kw_m["wing"] == "plate"
                    and kw_m["mu_scale"] == float(cars.get("mx5").mu_scale)
                    and c_s is s2.veh.car and kw_s is kw0 and n_s == "corsa")
        # the RACE page's Test: the bot ALONE in every car, in a pool, and
        # the result lands on the page (a 3 s drive each: the plumbing)
        s2._menu_show_race()
        test_row = "race_test" in [a for _, a in s2.menu.items]
        started = s2.start_bot_test(RACE_BOT_ANCHOR, T=3.0, workers=2)
        if started:
            s2._bot_test["res"].wait(180)
            s2._bot_test_poll()
        tr_ = s2.bot_test_result
        test_ok = (test_row and started and tr_ is not None
                   and [n for n, _ in tr_["rows"]] == list(RACE_BOT_CARS)
                   and all(not c.get("error") and c.get("s", 0.0) > 5.0 for _, c in tr_["rows"])
                   and s2.menu.sections[0][0].startswith("TEST"))
        s2.cancel_bot_test()
        s2.renderer = None
    grid_all = grid_ok and car_ok and pos_ok and moved2 and ghosts_ok and rows_ok and cyc2
    ok = (same and moved and gap_ok and hud_ok and rst_ok and page_ok and cyc_ok and stop_ok
          and grid_all and swcar_ok and test_ok)
    if verbose:
        print(f"  [V30] race vs bot: user car identical with/without bot {same}; "
              f"bot {bot_m:.0f} m / user {user_m:.0f} m in {T:.0f} s ({moved}); "
              f"gap {gs:+.2f} s {dm:+.0f} m ({gap_ok}); hud {hud_ok}; "
              f"reset {rst_ok}; menu page {page_ok} cycle {cyc_ok} stop {stop_ok}; "
              f"grid of 2 with a 540i: built {grid_ok} car {car_ok} slots {pos_ok} "
              f"moved {moved2} ghosts {ghosts_ok} page {rows_ok} car cycle {cyc2}; "
              f"swarm car {swcar_ok}; test in every car {test_ok}"
              f"  -> {'ok' if ok else 'FAIL'}")
    return ok, dict(same=same, progress=bot_m, gap_s=gs, gap_m=dm, grid=grid_all)


def _v31_records(tmp, verbose=True):
    """Lap records (drive/records.py): a scripted 3-lap headless run with a
    recorder on leaves the right top-5 file, re-simulating a lap from its
    controls log reproduces its time BIT FOR BIT, and a lap with a reset in
    it is not filed. The book lives in `tmp`: no player file is touched."""
    from . import records as recm
    root = os.path.join(tmp, "records")
    tr = trk.make_arena()
    drv = LapDriver(tr, margin=0.90)
    sim = _build("arena", driver=drv, start_V=25.0, gear=3, start_s=tr.length - 30.0)
    sim.s = tr.length - 30.0
    key = recm.class_key("arena", CAR_DEFAULT, "stock", "patch")
    book = recm.RecordBook(root)
    notes = []
    rec = recm.LapRecorder(book, key, dict(build_name="stock", build_json=None,
                                           assists=dict(abs=False, tc=False, steer_aid=True,
                                                        gearbox="auto")),
                           expect_global_wet=sim.global_wet, dt=sim.dt)
    rec.on_lap = lambda r: (sim._rec_lap(r), notes.append(sim._rec_msg))   # the session's
    sim.recorder = rec                                                      # own callback
    laps = []
    for _ in range(int(200.0 / sim.dt)):
        sim.step_physics(sim.dt)
        if rec.n_laps > len(laps):
            laps.append(rec.last)
            if len(laps) == 3:
                break
    rec.flush()                                 # the filing thread is done
    fresh = recm.RecordBook(root)               # read back from the FILE
    filed = fresh.laps(key)
    want = sorted(r["time"] for r in laps if r["valid"])
    timer = [e[3] for e in sim.events_log if e[0] == "lap"]
    path = book.path(key)
    file_ok = (os.path.exists(path) and len(laps) == 3 and all(r["valid"] for r in laps)
               and [lp["time"] for lp in filed] == want[:recm.TOP_N]
               and sorted(timer) == want            # the SAME floats the timer saw
               and all(len(lp["sectors"]) == len(tr.sector_s) for lp in filed)
               and fresh.pb_time(key) == want[0]
               and [r["pos"] for r in laps] == [1 + sorted([x["time"] for x in laps[:i + 1]]).index(r["time"])
                                                for i, r in enumerate(laps)])
    sizes = [recm.lap_size_bytes(lp) for lp in filed]
    size_ok = max(sizes) <= 300 * 1024
    rs = recm.resimulate(filed[len(filed) // 2])
    replay_ok = rs["exact"] and rs["sectors_exact"]
    # a lap with a reset in it is not a record
    for _ in range(int(3.0 / sim.dt)):
        sim.step_physics(sim.dt)
    sim.reset(to_checkpoint=True)
    n0 = rec.n_laps
    for _ in range(int(90.0 / sim.dt)):
        sim.step_physics(sim.dt)
        if rec.n_laps > n0:
            break
    rec.flush()
    reset_ok = (rec.n_laps == n0 + 1 and rec.last["valid"] is False and rec.last["why"] == "reset"
                and len(recm.RecordBook(root).laps(key)) == len(filed))
    # a LIVE setting change files the next lap under the new class (the
    # engine is in the key) with the new assists, and drops the lap in hand
    was = rec.recording
    sim.apply_setting("engine")
    sim.apply_setting("abs")
    st = sim.settings
    reset_ok = (reset_ok and was and not rec.recording
                and rec.key == recm.class_key("arena", st.car, st.engine, st.wet)
                and rec.meta["assists"]["abs"] == bool(st.abs) and rec._why == "engine changed")
    # the medal each lap earned (drive/medals.py, task 21) is on its note, and
    # the class's best is kept in its file
    try:
        from . import medals as mdl
        want_m = [mdl.medal_for(key, r["time"]) for r in laps]
        order = [m for m in mdl.MEDALS if m in want_m]
        medal_ok = (mdl.targets(key) is not None and any(want_m)   # never vacuous
                    and [r.get("medal") for r in laps] == want_m
                    and all((m or "").upper() in n for m, n in zip(want_m, notes))
                    and recm.RecordBook(root).load(key)["best_medal"] == (order[0] if order else None))
    except Exception as exc:               # noqa: BLE001
        want_m, medal_ok = [f"{type(exc).__name__}: {exc}"], False
    ok = file_ok and size_ok and replay_ok and reset_ok and medal_ok
    if verbose:
        print(f"  V31 records     : 3 laps {[round(r['time'], 3) for r in laps]} -> file "
              f"{[round(lp['time'], 3) for lp in filed]} ({file_ok}); "
              f"{max(sizes) / 1024:.0f} KB/lap max ({size_ok}); re-sim of "
              f"{rs['time'] if rs['time'] is None else round(rs['time'], 6)} s from the log, "
              f"bit for bit {rs['exact']}, sectors {rs['sectors_exact']}; reset mid-lap -> "
              f"not filed, a live engine / ABS change -> the new class and assists "
              f"({reset_ok}); medals {want_m}, best kept ({medal_ok})  -> {'ok' if ok else 'FAIL'}")
    return ok, dict(laps=[r["time"] for r in laps], kb=max(sizes) / 1024.0,
                    replay=rs["time"], exact=rs["exact"], notes=notes)


def _v32_prerace(tmp, verbose=True):
    """The pre-race screen as a menu flow, driven by events with no window
    (V26's style): it opens on RACE, one press races, EDIT goes to the
    garage, PICK lists the library's builds with their best time in this
    class and a pick restarts the session on it, the pause menu's Time trial
    reaches it, ESC backs out one page, and a click on a row fires it. And
    the screen is not offered to a scripted / headless run."""
    from types import SimpleNamespace
    from . import records as recm, prerace as prm
    root = os.path.join(tmp, "prerace")
    book = recm.RecordBook(root)
    key = recm.class_key("arena", CAR_DEFAULT, "sport", "patch")
    b_cur = dict(version=2, name="my corsa", mirror=True, builtin=False,
                 slots={"left": {"wing": "plate", "x": 0.97, "h": 0.9, "inc_deg": 0.0}})
    b_wet = dict(b_cur, name="wet setup", slots={"left": {"wing": "fin"}})
    js = {"my corsa": b_cur, "wet setup": b_wet}
    for t, name in ((61.40, "my corsa"), (62.10, "wet setup"), (60.95, "my corsa")):
        r = recm._fake_rec(t, [20.1, 20.4, 20.5])
        r["build"] = dict(name=name, json=js[name])
        book.insert(key, r)
    sim = _build("arena", driver=lambda t, v, T_: Controls())
    sim.renderer = SimpleNamespace(cfg=SimpleNamespace(mode="car_up"))
    sim.has_garage = True
    sim.recorder = recm.LapRecorder(book, key, {}, 1.0, sim.dt)
    sim.prerace = prm.PreRace(key, book, "my corsa", b_cur,
                              builds={"my corsa": b_cur, "wet setup": b_wet})
    ev = sim.handle_event

    def goto(action):
        acts = [a for _, a in sim.menu.items]
        i = acts.index(action)
        while sim.menu.idx != i:
            ev("nav_down")

    sim.run_headless(0.5)                          # the car has moved off the line
    sim.open_prerace()
    open_ok = (sim.menu.open and sim._menu_page == "prerace" and sim.paused
               and sim.menu.action() == "pr_race")
    ev("select")                                   # RACE: one press
    race_ok = (not sim.menu.open and not sim.paused and sim.s == 0.0
               and hypot(sim.veh.u, sim.veh.v) < 1e-9 and sim.lap.lap == 0
               and recm.RecordBook(root).last_build("arena")["name"] == "my corsa")
    ev("menu")                                     # pause menu -> Time trial
    goto("timetrial")
    ev("select")
    tt_ok = sim.menu.open and sim._menu_page == "prerace"
    goto("pr_pick")
    ev("select")
    items = sim.menu.items
    pick_ok = (sim._menu_page == "prerace_pick"
               and [a for _, a in items] == ["pr_build:my corsa", "pr_build:wet setup", "pr_back"]
               and "1:00.950" in items[0][0] and "1:02.100" in items[1][0])
    ev("menu")                                     # ESC: back to the pre-race page
    back_ok = sim._menu_page == "prerace" and sim.menu.open
    goto("pr_pick")
    ev("select")
    goto("pr_build:wet setup")
    ev("select")
    picked = (sim.quit and sim.stop_reason == "restart" and sim.prerace_pick is not None
              and sim.prerace_pick[0] == "wet setup" and sim.prerace_pick[1] == b_wet)
    sim.quit, sim.stop_reason, sim.prerace_pick = False, "", None
    sim.open_prerace()
    goto("pr_edit")
    ev("select")
    edit_ok = sim.quit and sim.stop_reason == "garage" and not sim.menu.open
    sim.quit, sim.stop_reason = False, ""
    sim.open_prerace()
    ev("menu")                                     # ESC on the pre-race: the pause page
    esc_ok = sim._menu_page == "main" and sim.menu.open
    ev("menu")
    # task 22: the Ghost 2 row steps the slot (LEFT / RIGHT or ENTER) and J
    # toggles the ghosts
    from .ghosts import GhostSet
    sim.ghosts = GhostSet(book, key, ref_fn=lambda k_: None)
    sim.open_prerace()
    goto("set:pr_ghost")
    row0 = sim.menu.items[sim.menu.idx][0]
    ev("nav_right")
    ghost_ok = (sim.ghosts.slot == "none" and "none" in sim.menu.items[sim.menu.idx][0]
                and "reference bot" in row0 and sim.menu.open)
    ev("select")
    ghost_ok = ghost_ok and sim.ghosts.slot == "top2" and sim.menu.open
    ev("menu"); ev("menu")                         # the page, then the pause page
    on0 = sim.ghosts.enabled
    ev("ghosts")                                   # J
    ghost_ok = ghost_ok and on0 and not sim.ghosts.enabled and not sim.menu.open
    ev("ghosts")
    sim.ghosts = None
    # the mouse: draw the page offscreen, click the RACE row
    mouse_ok = False
    try:
        import pygame
        pygame.font.init()
        surf = pygame.Surface((1280, 800))
        sim.run_headless(0.3)
        sim.open_prerace()
        goto("pr_edit")
        from .menu import CLICK_GUARD_DRAWS
        for _ in range(CLICK_GUARD_DRAWS):         # a quarter second of frames
            sim.menu.draw(surf)
        x, y = sim.menu.row_centre(0)
        ev(f"hover:{x}:{y}")
        hov = sim.menu.idx == 0
        ev(f"click:{x}:{y}")                       # the press arms RACE ...
        armed = sim.menu.open
        ev(f"release:{x}:{y}")                     # ... the release runs it
        mouse_ok = (hov and armed and not sim.menu.open and sim.s == 0.0
                    and not sim.paused)
    except Exception as exc:                       # noqa: BLE001
        print(f"    V32 mouse: {type(exc).__name__}: {exc}")
    # who gets the screen
    o = _Opts(headless=False, script=None, ml_drive=None, render=None)
    s_ = Settings(path="", track="arena")
    who_ok = (prm.wanted(o, s_) and not prm.wanted(_Opts(headless=True), s_)
              and not prm.wanted(_Opts(script="lap"), s_)
              and not prm.wanted(o, Settings(path="", track="dragstrip")))
    ok = all((open_ok, race_ok, tt_ok, pick_ok, back_ok, picked, edit_ok, esc_ok,
              mouse_ok, who_ok, ghost_ok))
    if verbose:
        print(f"  V32 pre-race    : opens on RACE {open_ok}; one press races {race_ok}; "
              f"Time trial {tt_ok}; pick lists builds + class bests {pick_ok}; "
              f"ESC back {back_ok}; pick restarts on it {picked}; EDIT -> garage {edit_ok}; "
              f"ESC -> pause page {esc_ok}; ghost-2 row + J {ghost_ok}; mouse hover + click "
              f"{mouse_ok}; scripted / headless / dragstrip skip it {who_ok}")
    return ok, dict(open=open_ok, race=race_ok, pick=pick_ok, mouse=mouse_ok)


#: V33's tolerance on the live delta against the lap's own trace: the trace's
#: time is stored to 1 ms and its distance to 1 mm, sampled at 50 Hz
V33_DELTA_TOL_S = 0.005


def _v33_ghost_delta(tmp, verbose=True):
    """Ghosts and the live delta (drive/ghosts.py): record one arena lap,
    re-simulate it from its controls log with a GhostSet whose PB IS that
    lap, and read the delta and the PB ghost's pose every 20 ms. The delta
    against a lap's own trace must stay within V33_DELTA_TOL_S of zero, the
    ghost must drive on top of the car, and every sector must flash purple
    (a lap's own sectors are the class's best). The book is in `tmp`."""
    from . import records as recm, ghosts as gh
    root = os.path.join(tmp, "ghosts")
    tr = trk.make_arena()
    sim = _build("arena", driver=LapDriver(tr, margin=0.90), start_V=25.0, gear=3,
                 start_s=tr.length - 30.0)
    sim.s = tr.length - 30.0
    key = recm.class_key("arena", CAR_DEFAULT, "stock", "patch")
    rec = recm.LapRecorder(recm.RecordBook(root), key, {}, sim.global_wet, sim.dt)
    sim.recorder = rec
    for _ in range(int(70.0 / sim.dt)):
        sim.step_physics(sim.dt)
        if rec.n_valid:
            break
    rec.close()
    book = recm.RecordBook(root)
    lap = book.pb(key)
    gs = gh.GhostSet(book, key, slot="none")
    deltas, gaps, flashes = [], [], []

    def on_step(s_):
        if gs._prog is None:               # the replay starts just past the line:
            gs.event(s_, ("start", 0, s_.lap.t_lap_start, float("nan")))   # the crossing
        if s_.n % 20 == 0:
            d = gs.delta(s_)
            if math.isfinite(d):
                deltas.append(d)
            g = gs.ghost_tuples(s_)
            if g:
                gaps.append(hypot(g[0][0] - s_.veh.x, g[0][1] - s_.veh.y))

    def on_event(s_, e):
        gs.event(s_, e)
        if e[0] == "sector":
            flashes.append(gs.flash_now(s_)[1])

    rs = recm.resimulate(lap, on_step=on_step, on_event=on_event)
    dmax = max((abs(d) for d in deltas), default=float("inf"))
    gmax = max(gaps, default=float("inf"))
    ok = (rs["exact"] and len(deltas) > 1000 and dmax <= V33_DELTA_TOL_S and gmax < 0.5
          and flashes == ["purple"] * len(tr.sector_s))
    if verbose:
        print(f"  V33 ghost delta : own lap {lap['time']:.3f} s re-driven, {len(deltas)} reads: "
              f"max |delta| {dmax * 1e3:.2f} ms (tol {V33_DELTA_TOL_S * 1e3:.0f} ms), PB ghost "
              f"within {gmax:.3f} m of the car, sector flashes {flashes}  -> {'ok' if ok else 'FAIL'}")
    return ok, dict(max_delta_s=dmax, max_gap_m=gmax, flashes=flashes)


#: V34's ceiling on the tutorial's sim time (it takes ~200 s)
V34_MAX_SIM_S = 900.0


def _v35_challenges(tmp, verbose=True):
    """Challenges (drive/challenges.py). ACHIEVABILITY: every challenge's
    reference run is driven again, headless, exactly as a session builds the
    car and with the meter attached through the Sim's own hooks; it must earn
    THREE stars, on a build that meets the challenge's rules and its 3-star
    efficiency bound, and give back the very value the file's thresholds were
    derived from (so the thresholds are the ones a measured run derives, not
    typed). And the page flow, by events with no window: ESC > Challenges
    lists them, a build that breaks a rule is refused with the reason and no
    Start, an allowed one starts (a restart in the challenge's class)."""
    from types import SimpleNamespace
    from . import challenges as chm
    from .aero.library import Library
    from .progress import Progress
    t_wall = time.perf_counter()
    lib = Library(os.path.join(tmp, "chal_lib"), use_xfoil=False)
    allc = chm.load_all()
    rows, all_ok, sim_s = [], len(allc) == 8, 0.0
    for cid, ch in allc.items():
        r = chm.measure(ch, lib)
        v, ref = r["value"], ch["ref"]["value"]
        same = math.isfinite(v) and abs(v - ref) <= 1e-9 * max(1.0, abs(ref))
        good = r["stars"] == 3 and not r["refusals"] and same
        all_ok = all_ok and good
        sim_s += r["t_sim"]
        rows.append(f"{cid} {chm.fmt_value(ch['goal']['metric'], v)} "
                    f"[{chm.stars_text(r['stars'])}]" + ("" if good else " FAIL"))
    # --- the page flow
    sim = _build("arena", driver=lambda t, v, T_: Controls())
    sim.renderer = SimpleNamespace(cfg=SimpleNamespace(mode="car_up"))
    sim.progress_file = Progress(os.path.join(tmp, "chal_progress.json"))
    top = dict(version=2, name="tall", mirror=True, builtin=False,
               slots={"left": {"wing": "plate"}, "top": {"wing": "rear-s1223", "x": -0.9,
                                                          "h": 1.55, "inc_deg": 6.0}})
    sim.challenge_build = (top, lib)
    ev = sim.handle_event

    def goto(action):
        i = [a for _, a in sim.menu.items].index(action)
        while sim.menu.idx != i:
            ev("nav_down")

    ev("menu")
    row_ok = "challenges" in [a for _, a in sim.menu.items]
    goto("challenges")
    ev("select")
    listed = [a for _, a in sim.menu.items if a.startswith("ch:")]
    list_ok = sim._menu_page == "challenges" and len(listed) == 8
    goto("ch:skid_dry")
    ev("select")
    refused = (sim._menu_page == "challenge" and "CANNOT START" in (sim.menu.note or "")
               and "top wing is not allowed" in sim.menu.note
               and not any(a.startswith("ch_go:") for _, a in sim.menu.items))
    ev("menu")                                     # ESC: back to the list
    back_ok = sim._menu_page == "challenges"
    sim.challenge_build = (chm.ref_build("plate"), lib)
    goto("ch:skid_dry")
    ev("select")
    goto("ch_go:skid_dry")
    ev("select")
    started = sim.quit and sim.stop_reason == "restart" and sim.challenge_pick == "skid_dry"
    flow_ok = row_ok and list_ok and refused and back_ok and started
    ok = all_ok and flow_ok
    wall = time.perf_counter() - t_wall
    if verbose:
        print(f"  V35 challenges  : {len(allc)} references, each 3 stars on its own rules and "
              f"value = the file's: {all_ok}; {', '.join(rows)}; pages: row {row_ok}, list "
              f"{list_ok}, a top wing refused with the reason {refused}, ESC {back_ok}, start "
              f"{started}; {sim_s:.0f} s sim in {wall:.1f} s  -> {'ok' if ok else 'FAIL'}")
    return ok, dict(rows=rows, sim_s=sim_s)


def _v34_tutorial(tmp, verbose=True):
    """The driving tutorial (drive/tutorial.py), end to end, headless: it is
    started from the pause menu's Tutorial page, every drive step is driven
    by ScriptedInput (the pedals, a weave, LapDriver through turn 1, round the
    skidpad and round the arena), R and F are pressed as events, every page
    is continued with ENTER, and when a step asks for another map the session
    is rebuilt there, as run_interactive_cli does -- one frame at a time,
    exactly as run_interactive runs it (events, 1/FPS of physics, the tick).
    Every drive step must PASS its own predicate (nothing skipped), the maps
    must go arena -> skidpad -> arena, and the progress file (a temporary
    one) must say done, with both wing circles and the lap measured."""
    from types import SimpleNamespace
    from . import tutorial as tu
    from .progress import Progress
    t_wall = time.perf_counter()
    path = os.path.join(tmp, "tutorial", "progress.json")
    prog = Progress(path)
    per = max(1, int(round(1.0 / (FPS * DT_PHYS))))
    box: dict = {}
    drivers: dict = {}

    def driver(t, veh, tr):
        tut = box.get("tut")
        sid = tut.step.id if (tut is not None and tut.active) else ""
        if sid == "pedals":
            return Controls(brake=1.0) if tut.mem.get("fast") else Controls(throttle=1.0)
        if sid == "steer":                         # the centreline, plus a weave
            pf = drivers.setdefault((id(tr), "weave"), PathFollower(9.0))
            c = pf(t, veh, tr)
            c.delta += 0.06 * sin(2.0 * math.pi * 0.4 * t)
            return c
        if sid in ("turn1", "reset", "wing_off", "wing_on", "lap"):
            k = (id(tr), sid == "wing_on")
            if k not in drivers:
                drivers[k] = LapDriver(tr, 0.85, wing_on=(sid == "wing_on"))
            return drivers[k](t, veh, tr)
        return Controls(brake=1.0)                 # a page: stand still

    def new_sim(track):
        s_ = _build(track, wing=("plate" if track == "skidpad" else "off"), driver=driver)
        s_.renderer = SimpleNamespace(cfg=SimpleNamespace(mode="car_up"))
        s_.progress_file = prog
        s_.tutorial = box.get("tut")
        return s_

    sim = new_sim("arena")
    ev = sim.handle_event

    def goto(action):
        i = [a for _, a in sim.menu.items].index(action)
        while sim.menu.idx != i:
            ev("nav_down")

    ev("menu")                                     # ESC > Tutorial > Start
    row_ok = "tutorial" in [a for _, a in sim.menu.items]
    goto("tutorial")
    ev("select")
    page_ok = sim._menu_page == "tutorial" and sim.menu.open
    goto("tut_start")
    ev("select")
    tut = box["tut"] = sim.tutorial
    start_ok = tut is not None and tut.step.id == "pedals" and not sim.menu.open
    maps, passed, pages, heads = [sim.track.name], [], [], set()
    sim_t, frames = 0.0, 0
    ids = [s_.id for s_ in tu.STEPS]
    while tut is not None and tut.active and sim_t < V34_MAX_SIM_S:
        if sim.quit:
            if sim.stop_reason != "restart":
                break
            sim_t += sim.t
            sim = new_sim(tut.map_wanted() or sim.track.name)
            maps.append(sim.track.name)
            continue
        sid = tut.step.id
        if sim.menu is not None and sim.menu.open:
            if sim._menu_page != "tutorial_step":
                break
            pages.append(sid)
            sim.inp.events.append("select")        # ENTER: Continue
        elif sid == "reset" and tut._t0 is not None and sim.t - tut._t0 > 2.0:
            sim.inp.events.append("reset")         # R
        elif sid == "wing_off" and sim.wing_on or sid == "wing_on" and not sim.wing_on:
            sim.inp.events.append("wing")          # F
        for e in sim.inp.poll_events():            # run_interactive's frame
            sim.handle_event(e)
        if not sim.paused:
            for _ in range(per):
                sim.step_physics(sim.dt)
        i0 = tut.i
        sim._tutorial_tick()
        if tut.i != i0 and tu.STEPS[i0].kind == "drive":
            passed.append(ids[i0])
        frames += 1
        if frames % 15 == 0 and tut.active and tut.step.kind == "drive":
            o = sim.hud_data().tutorial
            if o and o.get("head"):
                heads.add(tut.step.id if tut.step.title in o["head"] else "?")
    sim_t += sim.t
    drive_ids = [s_.id for s_ in tu.STEPS if s_.kind == "drive"]
    page_ids = [s_.id for s_ in tu.STEPS if s_.kind == "page"]
    sv = tu.saved_state(Progress(path))
    res = Progress(path).section("tutorial").get("results", {})
    off, on, lap = res.get("wing_off"), res.get("wing_on"), res.get("lap")
    measured = bool(off and on and lap and math.isfinite(off["ay"]) and off["ay"] > 0.3
                    and on["ay"] > 0.3 and 40.0 < lap["t"] < 120.0)
    ok = (row_ok and page_ok and start_ok and passed == drive_ids and pages == page_ids
          and maps == ["arena", "skidpad", "arena"] and tut is not None and tut.done
          and not tut.skipped and sv["done"] and measured and heads == set(drive_ids)
          and sim.tutorial is None)
    wall = time.perf_counter() - t_wall
    if verbose:
        print(f"  V34 tutorial    : menu row {row_ok}, page {page_ok}, started {start_ok}; "
              f"drive steps passed {len(passed)}/{len(drive_ids)}, pages {len(pages)}/"
              f"{len(page_ids)}, maps {' -> '.join(maps)}; wing off "
              f"{off['ay'] if off else float('nan'):.3f} g / on "
              f"{on['ay'] if on else float('nan'):.3f} g, lap "
              f"{lap['t'] if lap else float('nan'):.3f} s; done + saved {sv['done']}; "
              f"{sim_t:.0f} s sim in {wall:.1f} s  -> {'ok' if ok else 'FAIL'}")
        if not ok:
            print(f"    passed {passed}  pages {pages}  heads {sorted(heads)}  "
                  f"step {tut.step.id if tut else None}  mem {tut.mem if tut else None}")
    return ok, dict(passed=passed, maps=maps, results=res, sim_s=sim_t)


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
    s.car, s.ballast, s.ballast_at = "mx5", 75.0, "boot"
    s.save()
    back = Settings.load(path)
    rt_ok = (back.track, back.gearbox, back.abs, back.wet, back.camera,
             back.engine, back.tc, back.sound,
             back.car, back.ballast, back.ballast_at) == (
        "open", "clutch", False, "all", "car_up", "tuned", False, "low",
        "mx5", 75.0, "boot")
    bad = Settings(path=path)
    bad.track, bad.gearbox, bad.engine, bad.sound = "moon", "dsg", "v8", "11"
    bad.car, bad.ballast, bad.ballast_at = "delorean", 1e9, "roof"
    bad.clamp()
    clamp_ok = (bad.track, bad.gearbox, bad.engine, bad.sound, bad.car,
                bad.ballast, bad.ballast_at) == (
        "arena", "auto", ENGINE_DEFAULT, SOUND_DEFAULT, CAR_DEFAULT,
        cars.BALLAST_MAX, cars.BALLAST_DEFAULT)
    # the car library, through Settings: the default is the study's own car
    # and it is the SAME OBJECT, ballast really changes the CarSpec, and
    # power_scale is the Engine setting ALONE on every car (the car's engine
    # is engine_curve's business now, not power_scale's)
    d0 = Settings(path="")
    car_ok = (d0.car_spec() is cars.CORSA_C
              and d0.power_scale == ENGINE_SCALE[d0.engine]
              and Settings(path="", car="540i").power_scale
              == ENGINE_SCALE[ENGINE_DEFAULT])
    bal = Settings(path="", ballast=200.0, ballast_at="boot").car_spec()
    car_ok = car_ok and (bal.m == cars.CORSA_C.m + 200.0
                         and bal.wdist_f < cars.CORSA_C.wdist_f - 0.10
                         and bal.h_cg > cars.CORSA_C.h_cg
                         and bal.Izz > 1.4 * cars.CORSA_C.Izz)
    seat = Settings(path="", ballast=200.0, ballast_at="seat").car_spec()
    car_ok = car_ok and (seat.wdist_f == cars.CORSA_C.wdist_f
                         and seat.h_cg == cars.CORSA_C.h_cg
                         and seat.Izz == cars.CORSA_C.Izz)
    # the Engine labels must still read exactly as they did on the Corsa
    car_ok = car_ok and all(engine_hud(m) == ENGINE_HUD[m] for m in ENGINE_MODES)
    # ... and the whole session build, for every car, the way
    # _interactive_session does it: the CarSpec -> the tyre model, the
    # derived roll block, the powertrain. This is the one place that proves
    # a non-Corsa car reaches the physics intact, since the session loop
    # itself needs a window.
    from .tyre import CORSA_TYRE
    for name in CAR_MODES:
        for kg in (0.0, 150.0):
            sset = Settings(path="", car=name, ballast=kg, ballast_at="boot")
            cs = sset.car_spec()
            cv = Vehicle(cs, VehicleConfig(mu_scale=cs.mu_scale,
                                           power_scale=sset.power_scale))
            stock = cars.CARS[name]
            car_ok = car_ok and (
                cv.tyre.R0 == stock.tyre_R0 and cv.tyre_ref_ok
                and (cv.tyre is CORSA_TYRE) == (name == CAR_DEFAULT)
                and len(cv.pt_p.gear) == len(stock.gear)
                and cv.der.I_roll > 0.0 and cv._det_roll > 0.0
                and abs(cv.der.lltd_geo_f + cv.der.lltd_roll_f
                        - cv.cfg.roll_dist_f) < 1e-15
                and abs(sum(cv.Fz) - cs.m * G) < 1e-6
                and (cs is stock) == (kg == 0.0))
            cv.step(Controls(throttle=0.5, auto_gearbox=True), (1.0,) * 4,
                    (1.0,) * 4, DT_PHYS)
            car_ok = car_ok and all(f > 0.0 for f in cv.Fz)
    o = _Opts(track="skidpad", gearbox=None, abs=None, steer_limit=None,
              wet=None, camera=None, engine=None, tc=None, sound="off",
              car=None, ballast=None, ballast_at=None)
    back.apply_cli(o)
    # power_scale is the Engine setting alone, so 1.5 on every car
    cli_ok = (back.track == "skidpad" and back.gearbox == "clutch"
              and o.gearbox == "clutch" and o.auto_gearbox is False
              and o.abs is False and o.wet == "all" and o.steer_limit is True
              and o.engine == "tuned" and o.tc is False and o.sound == "off"
              and back.sound == "off" and back.car == "mx5"
              and o.car == "mx5" and o.ballast == 75.0 and o.ballast_at == "boot"
              and back.power_scale == 1.5
              and Settings(path="", engine="tuned").power_scale == 1.5)
    # an EXPLICIT --car / --ballast still wins over the file
    o2 = _Opts(track=None, gearbox=None, abs=None, steer_limit=None, wet=None,
               camera=None, engine=None, tc=None, sound=None,
               car="540i", ballast=125.0, ballast_at="nose")
    b2 = Settings.load(path).apply_cli(o2)
    cli_ok = cli_ok and (b2.car, b2.ballast, b2.ballast_at) == ("540i", 125.0, "nose")
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

    def goto(action: str) -> int:
        """Put the cursor on the row with this action, wherever it is.

        The page grew Car / Ballast / Ballast at rows, and a test that walks
        it by counting nav_downs has to be rewritten every time a row is
        added -- which is how a settings test stops testing settings. Look
        the row up instead."""
        tgt = [a for _, a in sim._settings_items()].index(action)
        assert sim.menu.open and sim._menu_page == "settings", "not on the page"
        while sim.menu.idx < tgt:
            ev("nav_down")
        while sim.menu.idx > tgt:
            ev("nav_up")
        return tgt
    i_eng = goto("set:engine")
    ev("select")                                    # sport -> stock (wraps)
    eng_ok = (st.engine == ENGINE_MODES[(ENGINE_MODES.index(ENGINE_DEFAULT) + 1)
                                         % len(ENGINE_MODES)]
              and veh.cfg.power_scale == ENGINE_SCALE[st.engine]
              and abs(veh.pt_p.T_clutch_cap - 200.0 * veh.cfg.power_scale) < 1e-9
              and sim.menu.idx == i_eng)
    goto("set:gearbox")
    ev("select")                                    # auto -> manual
    c = inp.update(DT_PHYS, 0.0)
    gb_ok = (st.gearbox == "manual" and sim.gearbox == "manual"
             and (c.auto_gearbox, c.auto_clutch) == (False, True)
             and sim.menu.open and sim._menu_page == "settings")
    ev("select")                                    # manual -> clutch: neutral
    neut_ok = (st.gearbox == "clutch" and veh.gear == 0 and veh.pt_s.gear == 0)
    for _ in range(300):
        veh.step(inp.update(DT_PHYS, 0.0), (1.0,) * 4, (1.0,) * 4, DT_PHYS)
    neut_ok = neut_ok and not veh.stalled and veh.rpm > 600.0
    ev("select")                                    # clutch -> auto
    ev("select")                                    # auto -> manual
    gb_ok = gb_ok and neut_ok and st.gearbox == "manual"
    i_abs = goto("set:abs")
    ev("select")                                    # on -> off
    abs_ok = (st.abs is False and veh.cfg.abs_on is False and sim.menu.idx == i_abs)
    i_tc = goto("set:tc")
    ev("select")                                    # on -> off
    tc_ok = (st.tc is False and veh.cfg.tc_on is False and sim.menu.idx == i_tc)
    ev("select")                                    # back on
    tc_ok = tc_ok and st.tc is True and veh.cfg.tc_on is True
    goto("set:steer_aid")
    ev("select")
    aid_ok = (st.steer_aid is False and inp.kb.steer_limit is False)
    ev("select")                                    # back on again
    goto("set:camera")
    ev("select")
    cam_ok = st.camera == "chase" and sim.renderer.cfg.mode == "chase"
    i_snd = goto("set:sound")
    ev("select")                                    # mid -> high (no window: no CarSound)
    snd_ok = (st.sound == SOUND_MODES[(SOUND_MODES.index(SOUND_DEFAULT) + 1)
                                       % len(SOUND_MODES)]
              and sim.audio is None and sim.menu.idx == i_snd)
    # the three new rows: each one demands a session rebuild, because a
    # different CarSpec is a different tyre model, roll block and gearbox
    row_ok = True
    for key, want in (("set:car", CAR_MODES[1]),
                      ("set:ballast", cars.BALLAST_KG[1]),
                      ("set:ballast_at", cars.BALLAST_STATIONS[
                          (cars.BALLAST_STATIONS.index(cars.BALLAST_DEFAULT) + 1)
                          % len(cars.BALLAST_STATIONS)])):
        sim.quit, sim.stop_reason = False, ""
        goto(key)
        ev("select")
        got = {"set:car": st.car, "set:ballast": st.ballast,
               "set:ballast_at": st.ballast_at}[key]
        row_ok = row_ok and got == want and sim.stop_reason == "restart" and sim.quit
        sim.quit, sim.stop_reason = False, ""
        ev("menu"); ev("nav_down"); ev("select")    # the page again
    st.car, st.ballast, st.ballast_at = CAR_DEFAULT, 0.0, cars.BALLAST_DEFAULT
    st.save()
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
    # browse before committing: LEFT / RIGHT on a restart row only change the
    # row (and what the file says stays the running value); ESC drops the
    # browse; ENTER on a browsed row applies it and restarts
    ev("menu"); ev("nav_down"); ev("select")        # settings
    goto("set:track")
    ev("nav_right")                                 # skidpad -> dragstrip, preview
    prev_ok = (st.track == "dragstrip" and not sim.quit and sim.menu.open
               and sim._pending == {"track": "skidpad"}
               and "ENTER applies" in sim._settings_items()[sim.menu.idx][0])
    goto("set:abs")
    ev("select")                                    # a live change saves: not the preview
    prev_ok = prev_ok and Settings.load(path).track == "skidpad" and st.track == "dragstrip"
    goto("set:track")
    ev("nav_left")                                  # back where it was: nothing pending
    prev_ok = prev_ok and st.track == "skidpad" and not sim._pending
    ev("nav_right"); ev("nav_right")                # dragstrip -> arena (wraps)
    goto("set:car")
    ev("nav_left")                                  # corsa -> 540i (wraps back)
    ev("menu")                                      # ESC: the browse is dropped
    prev_ok = (prev_ok and st.track == "skidpad" and st.car == CAR_DEFAULT
               and not sim._pending and sim._menu_page == "main" and not sim.quit)
    ev("select"); goto("set:track")                 # the cursor is on Settings
    ev("nav_right"); ev("nav_left"); ev("nav_left")  # -> open
    ev("select")                                    # ENTER applies -> restart
    prev_ok = (prev_ok and st.track == "open" and sim.quit
               and sim.stop_reason == "restart" and not sim.menu.open
               and not sim._pending and Settings.load(path).track == "open")
    goto_back = st.track
    sim.quit, sim.stop_reason = False, ""
    ev("menu"); ev("nav_down"); ev("select"); goto("set:gearbox")
    ev("nav_left")                                  # LEFT on a live row: manual -> auto
    prev_ok = prev_ok and st.gearbox == "auto" and sim.gearbox == "auto" and sim.menu.open
    ev("nav_right")                                 # and back
    prev_ok = prev_ok and st.gearbox == "manual" and goto_back == "open"
    ev("menu"); ev("menu")
    sim.quit, sim.stop_reason = False, ""
    ev("garage")
    gar_ok = sim.stop_reason == "garage" and sim.quit
    sim.quit, sim.stop_reason = False, ""
    ev("menu"); ev("nav_down"); ev("select")        # settings
    tgt = [a for _, a in sim._settings_items()].index("garage")
    while sim.menu.idx < tgt:
        ev("nav_down")
    ev("select")
    gar2_ok = sim.stop_reason == "garage" and sim.quit and not sim.menu.open
    ok = all((rt_ok, clamp_ok, cli_ok, cycle_ok, car_ok, m_open, page_ok,
              eng_ok, gb_ok, abs_ok, tc_ok, aid_ok, cam_ok, snd_ok, row_ok,
              saved_ok, back_ok, closed_ok, map_ok, tab_ok, prev_ok, gar_ok,
              gar2_ok))
    if verbose:
        print(f"  V26 settings    : round-trip {rt_ok}, clamp {clamp_ok}, cli {cli_ok}, "
              f"cycle {cycle_ok}; menu open {m_open}, settings page {page_ok}, "
              f"engine {eng_ok}, gearbox {gb_ok}, abs {abs_ok}, tc {tc_ok}, aid {aid_ok}, "
              f"camera {cam_ok}, sound {snd_ok}, saved {saved_ok}, back {back_ok}, "
              f"closed {closed_ok}, map restart {map_ok}, TAB {tab_ok}, "
              f"browse {prev_ok}, garage {gar_ok}/{gar2_ok}; car/ballast {car_ok}, "
              f"rows {row_ok}")
    return ok, dict(round_trip=rt_ok, cli=cli_ok, car=car_ok and row_ok,
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
                     ("V30", lambda: _v30_race_vs_bot(verbose)),
                     ("V31", lambda: _v31_records(tmp, verbose)),
                     ("V32", lambda: _v32_prerace(tmp, verbose)),
                     ("V33", lambda: _v33_ghost_delta(tmp, verbose)),
                     ("V34", lambda: _v34_tutorial(tmp, verbose)),
                     ("V35", lambda: _v35_challenges(tmp, verbose)),
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
F flank-wing toggle | G cycle wing side (auto / left / right / both = air brake)
R reset to last sector line | SHIFT+R full reset
P pause | O single physics step | [ ] slow-mo 0.25x / 1.0x
C camera | - / = zoom | 0 auto zoom | H HUD | V vectors | B g-g | N skid | X clear
T toggle wet | M telemetry marker | L toggle recording | K arm a SEED LAP for the swarm | TAB next map | J ghosts
BACKSPACE garage (3D panel editor) | ESC menu: settings (map, engine, gearbox, ABS, TC, aids, sound), reset, race vs bot, quit
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
    p.add_argument("--car", default=None, choices=CAR_MODES,
                   help="which car (default corsa: the study's own, and the "
                        "fixed default for scripts and --headless)")
    p.add_argument("--ballast", type=float, default=None,
                   help=f"kg of added mass, 0-{cars.BALLAST_MAX:.0f}")
    p.add_argument("--ballast-at", dest="ballast_at", default=None,
                   choices=cars.BALLAST_STATIONS,
                   help="where the ballast sits (default %s)" % cars.BALLAST_DEFAULT)
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
    p.add_argument("--dev-flank", dest="dev_flank", default="outer",
                   choices=("outer", "inner"),
                   help="which flank the panel deploys on (VehicleConfig.dev_flank). "
                        "'outer' is the study's configuration; 'inner' is the "
                        "measurement in .handoff/14-inner-flank.md and needs a panel "
                        "whose section is turned over to keep the force inboard")
    p.add_argument("--build", default=None,
                   help="drive a car saved in the garage library (runs/library/builds/NAME.json)")
    p.add_argument("--ml-drive", default=None, metavar="CHECKPOINT",
                   help="put a trained drive.ml policy in the driver's seat "
                        "(e.g. drive/ml/checkpoints/arena_plate.json); the "
                        "sub-package is optional and is imported only here")
    p.add_argument("--race", default=None, metavar="BOT[,BOT2[,BOT3]]",
                   help="race against the ML driver: 'anchor' (the built-in driver), "
                        "'best' (the newest swarm checkpoint) or a checkpoint path, "
                        "comma-separated for up to 3 bots on the grid; also "
                        "ESC > Race vs bot")
    p.add_argument("--race-car", dest="race_car", default=None, metavar="CAR[,CAR2[,CAR3]]",
                   help="what each bot drives: 'same' (your car, the default) or a "
                        "stock corsa / mx5 / 540i, per bot like --race")
    p.add_argument("--garage", action="store_true",
                   help="open the 3D editor first; ENTER / cross drives the "
                        "car you built, BACKSPACE / touchpad comes back")
    p.add_argument("--seed-lap", dest="seed_lap", action="store_true",
                   help="start with the seed-lap recorder ARMED (the K key): the "
                        "next complete valid lap is saved to runs/swarm/ for --swarm-seed")
    p.add_argument("--swarm", type=int, default=None, metavar="N",
                   help="deploy a swarm of N learning cars on this map and car "
                        "(drive.ml.swarm, a genetic algorithm); the window replays "
                        "each generation while the next one is computed")
    p.add_argument("--swarm-seed", dest="swarm_seed", default="none",
                   help="'none' (the anchor driver), a seed lap the user drove "
                        "(runs/swarm/seed_*.json), or a Policy checkpoint to breed from")
    p.add_argument("--swarm-gens", dest="swarm_gens", type=int, default=0,
                   help="stop auto-running after this many generations (0 = until ESC)")
    p.add_argument("--swarm-T", dest="swarm_T", type=float, default=70.0,
                   help="s of sim per car per generation")
    p.add_argument("--swarm-name", dest="swarm_name", default=None,
                   help="name of the run (runs/swarm/<name>_state.json, "
                        "drive/ml/checkpoints/swarm_<name>.json)")
    p.add_argument("--swarm-resume", dest="swarm_resume", default=None,
                   help="continue a swarm from its runs/swarm/*_state.json")
    p.add_argument("--swarm-car", dest="swarm_car", default=None,
                   choices=("same",) + tuple(cars.CAR_ORDER),
                   help="what the swarm breeds in: 'same' (your car, the default) or "
                        "a stock corsa / mx5 / 540i on your settings, like --race-car")
    p.add_argument("--swarm-fast", dest="swarm_fast", action="store_true",
                   help="no replay: the next generation starts the moment one is "
                        "scored (V in the swarm window toggles it)")
    p.add_argument("--swarm-save", dest="swarm_save", default="ask",
                   choices=("ask", "always", "never"),
                   help="whether ESC in the swarm window writes the best as a "
                        "checkpoint: ask (ENTER save / ESC discard), always, or "
                        "never (K still saves)")
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
        # The car is a fixed default too (CONTRACT section 8): every
        # acceptance number is the stock Corsa C with no ballast. An
        # EXPLICIT --car / --ballast does reach a script, the way --wing
        # does -- the car is the subject of the measurement, not a driver
        # aid, and measuring another one with the repo's own rigs is the
        # point of having them. Nothing in validate.py passes either flag.
        opts.car = opts.car or CAR_DEFAULT
        opts.ballast = 0.0 if opts.ballast is None else float(opts.ballast)
        opts.ballast_at = opts.ballast_at or cars.BALLAST_DEFAULT
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
                     start_V=20.0 if opts.track != "dragstrip" else 0.0, gear=3,
                     car=_opts_car(opts))
        sim.run_headless(opts.duration)
        if sim.telem:
            sim.telem.close()
        print(f"headless {opts.duration:g} s on {opts.track}: "
              f"t={sim.t:.3f} s, laps={sim.lap.lap}, "
              f"best={sim.lap.best_lap:.3f} s, csv={opts.telemetry}")
        return 0

    # ---- the swarm ------------------------------------------------
    if opts.swarm is not None or opts.swarm_resume:
        return run_swarm_cli(opts)

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
        opts.mass_points = ()
        opts.build_name, opts.build_json = f"{design.wing} panel", None
        return kw
    lib = lib or grg.library()
    kw = design.cfg_kwargs(lib)
    opts.wing, opts.wing_x, opts.wing_h = kw["wing"], kw["x_w"], kw["h_w"]
    opts.wing_inc = degrees(kw["delta_dev_geom"])
    opts.wing_cfg = kw
    opts.hud_cfg = design.hud_kwargs(lib)
    # the three fitted wings' MASS, charged to the car at their own stations
    # (drive/aero/wing.wing_mass -> CarBuild.mass_points). Empty slots give
    # an empty tuple, so a car with no wings is the car every rig measures.
    opts.mass_points = design.mass_points(lib)
    #  what a lap record files as its build (drive/records.py)
    opts.build_name, opts.build_json = design.name, design.to_json()
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
    grg, design, lib = _resolve_design(opts)

    mode = "garage" if (opts.garage and grg is not None) else "drive"
    pad = None
    #  the build a map opens with is the one last USED there (the pre-race
    #  screen, drive/prerace.py): at launch -- unless --build / --wing named
    #  one -- and whenever the map changes; never over a car the garage has
    #  just built. The switch is in memory: runs/garage_design.json is the
    #  garage's working car and only the garage writes it
    seen_track = None
    explicit = bool(getattr(opts, "build", None)) or any(
        a.startswith("--wing") for a in sys.argv[1:])
    #  the driving tutorial (drive/tutorial.py): runs/progress.json is a
    #  player file, opened by a player session only; offered on its first launch
    opts.progress, opts.tutorial_offer = None, False
    if _player_session(opts):
        try:
            from .progress import Progress
            from .tutorial import saved_state
            opts.progress = Progress()
            for n in opts.progress.notes:
                print(f"progress: {n}")
            opts.tutorial_offer = not saved_state(opts.progress)["offered"]
        except Exception as exc:           # noqa: BLE001 -- never stops a drive
            print(f"progress unavailable ({type(exc).__name__}: {exc})")
            opts.progress = None
    try:
        while True:
            from_garage = False
            if mode == "garage":
                g = grg.Garage((w, h), design, pad=pad, lib=lib)
                #  the wing-design tutorial (drive/wing_tutorial.py) rides
                #  on opts across the garage <-> drive round trips
                g.progress = getattr(opts, "progress", None)
                g.tutor = getattr(opts, "wing_tutor", None)
                action = g.run()
                design, pad = g.build, g.pad
                t_ = g.tutor
                opts.wing_tutor = t_ if (t_ is not None and t_.active) else None
                if action != "drive":
                    break
                design.save()
                _apply_design(opts, design, lib)
                print(f"garage -> drive: {design.summary(lib)}")
                mode = "drive"
                from_garage = True
            tut = getattr(opts, "tutorial", None)
            if tut is not None and not tut.active:
                tut = opts.tutorial = None
            if tut is not None and tut.map_wanted() and settings.track != tut.map_wanted():
                settings.track = tut.map_wanted()   # the step's map: TAB cannot leave it
                settings.save()
            #  a challenge (drive/challenges.py) runs in its class; a class
            #  changed from the settings page ends it
            chal = getattr(opts, "challenge", None)
            if chal is not None and _challenge_class(settings) != chal["class"]:
                print(f"challenge '{chal['title']}': the class changed; it is over")
                _challenge_restore(opts, settings, keep_changed=chal["class"])
                chal = None
            if (grg is not None and design is not None and not from_garage
                    and chal is None
                    and settings.track != seen_track
                    and not (explicit and seen_track is None)):
                d2 = _track_build(grg, lib, design, settings.track, opts)
                if d2 is not None:
                    design = d2
                    _apply_design(opts, design, lib)
            seen_track = settings.track
            #  the tutorial's wing laps need a flank wing: a car without one
            #  drives them with the library's plate (in memory, never saved)
            tut_car = None
            if tut is not None and tut.wants_wing():
                from .tutorial import wing_car
                tut_car = wing_car(grg, design, lib)
            if tut_car is not None:
                _apply_design(opts, tut_car, lib)
            elif getattr(opts, "tutorial_car", False) and design is not None:
                _apply_design(opts, design, lib)   # the player's own car back
            opts.tutorial_car = tut_car is not None
            if grg is not None and design is not None and tut_car is None:
                _track_build_used(settings.track, design, opts)
            sim = _interactive_session(opts, pad=pad, settings=settings,
                                       garage=(grg is not None))
            t_ = getattr(sim, "tutorial", None)    # started, or still running
            opts.tutorial = t_ if (t_ is not None and t_.active) else None
            _challenge_switch(sim, opts, settings)
            pad = getattr(sim.inp, "pad", pad)
            opts.race_menu = dict(sim.race_opts, active=bool(sim.rivals))
            gs = getattr(sim, "ghosts", None)
            if gs is not None:             # the ghost slot and J survive a restart
                opts.ghost_slot, opts.ghosts_on = gs.slot, gs.enabled
            pick = getattr(sim, "prerace_pick", None)
            if pick and grg is not None:
                #  the pre-race PICK: that saved build is the car from now on;
                #  a car being driven that is in no library file is saved as
                #  one first, so a pick never loses it
                _autosave_build(design, lib)
                design = grg.CarBuild.from_json(pick[1])
                design.clamp(lib)
                _apply_design(opts, design, lib)
                print(f"pre-race: driving the build '{pick[0]}'")
            if getattr(sim, "wing_tutor_start", False) and grg is not None:
                from .wing_tutorial import WingTutor, saved_state
                opts.wing_tutor = WingTutor(opts.progress,
                                            start=saved_state(opts.progress)["step"])
            if sim.stop_reason == "garage" and grg is not None:
                mode = "garage"
                continue
            if sim.stop_reason == "swarm":
                # the pause menu's Deploy: the swarm runs in THIS window on
                # this session's car and map, then ESC brings a new session
                # back here with the same settings
                launch = sim.swarm_launch or dict(SWARM_MENU_DEFAULTS)
                opts.swarm = int(launch["pop"])
                opts.swarm_seed = launch["seed"]
                opts.swarm_gens = int(launch["gens"])
                opts.swarm_T = float(launch["T"])
                opts.swarm_fast = launch.get("view", "replay") == "fast"
                opts.swarm_save = launch.get("save", "ask")
                opts.swarm_car = launch.get("car", "same")
                opts.swarm_resume = None
                opts.swarm_name = None
                opts.swarm_saved = None
                try:
                    run_swarm_cli(opts, settings=settings, embedded=True)
                except Exception as exc:
                    print(f"swarm: {type(exc).__name__}: {exc}")
                saved = getattr(opts, "swarm_saved", None)
                if saved and os.path.exists(saved):
                    #  the bot just saved is bot 1 on the RACE page, in the
                    #  car it was bred in: racing it is ESC > Race > Start
                    rm = dict(getattr(opts, "race_menu", None) or {})
                    rm["bot"] = saved
                    rm["car"] = opts.swarm_car or "same"
                    opts.race_menu = rm
                    print(f"race it: ESC > Race vs bot > Start "
                          f"(bot 1 is now {race_bot_label(saved)})")
                opts.swarm = None
                opts.swarm_car = None
                opts.swarm_saved = None
                opts.swarm_menu = dict(launch)     # the page remembers its values
                opts.prerace_skip = True           # back from the swarm: drive, not a timed start
                continue
            if sim.stop_reason == "restart":
                continue
            break
    finally:
        _challenge_restore(opts, settings)     # quit mid-challenge: the player's class back
        pygame.quit()
    return 0


# ==================================================================== #
#  THE SWARM  (drive.ml.swarm behind a window)                          #
# ==================================================================== #
#: playback speeds the [ ] keys step through
SWARM_SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0)
#: the window's frame rate with the replay OFF: the pool has the cores
SWARM_FAST_FPS = 12
#: generations of history the table shows with the replay off
SWARM_TABLE_ROWS = 12
#: ghost colours: the ranked population from best (green) to worst (red),
#: a crashed car's last pose (grey), the user's own seed lap (cyan)
C_SWARM_BEST = (90, 200, 110)
C_SWARM_WORST = (200, 80, 70)
C_SWARM_DEAD = (95, 95, 95)
C_SWARM_USER = (80, 200, 230)


def _lerp3(a, b, t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


class _Replay:
    """One car's rollout trace, (t, x, y, psi, u, wing_deploy, wing_side,
    top_deploy) every 20 training steps, interpolated at any playback time.
    A 7-column trace (a state file from before the top wing was recorded)
    is padded with a stowed top wing."""

    __slots__ = ("arr", "t_end", "alive_to_end", "colour", "label")

    def __init__(self, trace, colour, alive_to_end: bool, label: str = ""):
        a = np.asarray(trace, dtype=np.float64)
        if a.ndim != 2 or len(a) == 0:
            a = np.zeros((1, 8))
        elif a.shape[1] < 8:
            a = np.hstack([a, np.zeros((len(a), 8 - a.shape[1]))])
        self.arr = a
        self.t_end = float(a[-1, 0])
        self.alive_to_end = bool(alive_to_end)
        self.colour = colour
        self.label = label

    def pose(self, tau: float):
        """(x, y, psi, u, wing_deploy, wing_side, alive, top_deploy) at
        playback time tau."""
        a = self.arr
        if tau >= self.t_end:
            r = a[-1]
            return r[1], r[2], r[3], r[4], r[5], int(r[6]), self.alive_to_end, r[7]
        i = int(np.searchsorted(a[:, 0], tau, side="right"))
        if i <= 0:
            r = a[0]
            return r[1], r[2], r[3], r[4], r[5], int(r[6]), True, r[7]
        r0, r1 = a[i - 1], a[i]
        f = (tau - r0[0]) / max(r1[0] - r0[0], 1e-9)
        psi = r0[3] + _wrap_pi(r1[3] - r0[3]) * f
        return (r0[1] + (r1[1] - r0[1]) * f, r0[2] + (r1[2] - r0[2]) * f, psi,
                r0[4] + (r1[4] - r0[4]) * f, r0[5] + (r1[5] - r0[5]) * f,
                int(r1[6]), True, r0[7] + (r1[7] - r0[7]) * f)


def _swarm_seed_spec(spec: str | None) -> str:
    """'none' | a path that exists | 'none' with a note. ('latest' and
    'best' are resolved to a path by `run_swarm_cli` before this.)"""
    if not spec or spec == "none":
        return "none"
    if not os.path.exists(spec):
        print(f"--swarm-seed {spec}: not found; breeding from the anchor driver")
        return "none"
    return spec


def run_swarm_cli(opts, settings=None, embedded: bool = False) -> int:
    """`--swarm N`: a population of N learning cars on the session's map and
    car. The window REPLAYS a scored generation as ghost cars (the best one is
    drawn as the car and carries the camera) while the pool is already
    computing the next; the user's own seed lap, if one was given, drives
    alongside in cyan. Keys:

        SPACE  auto-run on / off        ENTER  next generation now
        K      name + save the best   [ ]    playback speed
        V      replay on / off          C      camera
        - = 0  zoom                     ESC    quit

    With the replay OFF (`--swarm-fast`, the page's Replay row, or V) the
    next generation starts the moment one is scored and the workers send
    no trace back: the pool is the only clock. ESC then does what
    `--swarm-save` (the page's Save best row) says: ask (ENTER saves under
    the name typed, ESC discards -- the population is in the state file
    either way), always, or never.

    `K` writes `drive/ml/checkpoints/swarm_<name>.json` -- a plain `Policy`
    that the RACE page lists, `--ml-drive` drives and a later `--swarm-seed`
    breeds from -- with the best lap RE-MEASURED at the contract's 1 ms step;
    ESC with nothing saved yet asks, saves or discards per `--swarm-save`.
    The whole population is saved to `runs/swarm/<name>_state.json` after
    every generation and `--swarm-resume` continues it. drive.ml is imported
    HERE and nowhere else on this path.
    """
    try:
        import pygame
    except Exception as exc:
        print(f"pygame unavailable ({exc}); use `python3 -m drive.ml.swarm` headless")
        return 1
    try:
        from .ml.swarm import Swarm
        from .ml import clone as ml_clone, swarm as ml_swarm
    except Exception as exc:
        print(f"--swarm: drive.ml unavailable ({type(exc).__name__}: {exc})")
        return 1
    from types import SimpleNamespace

    if not embedded:
        pygame.init()
        settings = Settings.load().apply_cli(opts)
        settings.save()
        _resolve_design(opts)
    w, h = (int(v) for v in opts.size.lower().split("x"))
    tr, car, cfg_kwargs, global_wet = _session_car(opts, settings)
    #  what it breeds in: the session's car, or a stock one (--swarm-car,
    #  the page's Car row); a resume with no car given keeps the one the
    #  swarm was bred in
    want_car = getattr(opts, "swarm_car", None) or "same"
    if opts.swarm_resume and want_car == "same":
        want_car = _swarm_state_car(opts.swarm_resume, settings.car)
    car, cfg_kwargs, car_name = _swarm_car(want_car, car, cfg_kwargs, settings.car)
    stock = want_car not in (None, "", "same") and car_name == want_car
    if global_wet != 1.0:
        #  the rollout has no global wet; the same physics reached through
        #  the config's grip scale (vehicle.py: mu[i] * cfg.mu_scale)
        cfg_kwargs = dict(cfg_kwargs, mu_scale=cfg_kwargs["mu_scale"] * global_wet)
    track_kw = dict(radius=opts.radius, cw=opts.cw, surfaces=(opts.wet != "none"))
    car_title = cars.car_name(car_name) + (" (stock)" if stock else "")
    opts.swarm_saved = None         # the checkpoint this run writes, if any

    user_replay = None
    user_lap = None
    if opts.swarm_seed == "best":
        import glob
        c = sorted(glob.glob(os.path.join("drive", "ml", "checkpoints", "swarm_*.json")),
                   key=os.path.getmtime)
        opts.swarm_seed = c[-1] if c else "none"
        if not c:
            print("swarm seed 'best': no swarm checkpoint saved yet; breeding from the anchor")
    if opts.swarm_seed == "latest":
        #  the newest seed lap ON THIS MAP: a lap from another circuit would
        #  be refused by the clone anyway, and the menu labels it the same way
        import glob
        c = sorted((p for p in glob.glob(os.path.join(SEED_LAP_DIR, "seed_*.json"))
                    if f"seed_{tr.name}_" in os.path.basename(p)), key=os.path.getmtime)
        opts.swarm_seed = c[-1] if c else "none"
        if not c:
            print(f"swarm seed 'latest': no seed lap recorded on {tr.name} "
                  f"(ESC > Deploy swarm > Drive seed lap); breeding from the anchor driver")
    if opts.swarm_resume:
        sw = Swarm.load_state(opts.swarm_resume, car=car, cfg_kwargs=cfg_kwargs)
        print(f"swarm: resumed {sw.name} at generation {sw.gen} "
              f"({len(sw.history)} scored)")
        if sw.seed_source.startswith("lap:"):
            cand = os.path.join(SEED_LAP_DIR, sw.seed_source[4:])
            if os.path.exists(cand):
                seed = ml_clone.load_seed_lap(cand)
                user_replay = _Replay(ml_clone.seed_lap_trace(seed), C_SWARM_USER, True, "you")
                user_lap = seed.get("lap_time")
    else:
        spec = _swarm_seed_spec(opts.swarm_seed)
        sw = Swarm(pop=int(opts.swarm), track=tr.name, wing=opts.wing, car=car,
                   cfg_kwargs=cfg_kwargs, T=opts.swarm_T, name=opts.swarm_name,
                   track_kw=track_kw, car_name=car_name)
        try:
            sw.seed_from(spec, tr)
        except Exception as exc:
            print(f"--swarm-seed {spec}: {type(exc).__name__}: {exc}; "
                  f"breeding from the anchor driver")
            sw.seed_from("none", tr)
        if sw.seed_source.startswith("lap:"):
            seed = ml_clone.load_seed_lap(spec)
            user_replay = _Replay(ml_clone.seed_lap_trace(seed), C_SWARM_USER, True, "you")
            user_lap = seed.get("lap_time")
        print(f"swarm {sw.name}: {sw.pop} cars, seed {sw.seed_source}, "
              f"{tr.title or tr.name}, {car_title} {car.m:.0f} kg, "
              f"wing {opts.wing}, T {sw.T:.0f} s, {sw.workers} workers")

    from . import render as rnd
    rnd.set_car(car)
    cfgv = rnd.ViewConfig(size=(w, h), fps=opts.fps, mode=settings.camera or "car_up")
    cfgv.hud = "off"
    cfgv.show_vectors = False
    cfgv.show_gg = False
    renderer = rnd.Renderer(cfgv, tr, headless=(opts.render == "offscreen"))
    print("swarm: SPACE auto-run | ENTER next generation | K name + save best | V replay | "
          "[ ] speed | C camera | - = 0 zoom | ESC quit")

    replays: list = []
    shown = None                    # the history row of the generation on screen
    tau = 0.0
    speed_i = 1
    autorun = True
    saved_path = None
    saved_measured = None
    note, note_until = "", 0.0
    measuring = None                # (AsyncResult, pool) while re-measuring the best
    gens_target = int(opts.swarm_gens or 0)
    gens_done_at_start = len(sw.history)
    clock = pygame.time.Clock()
    running = True
    want_next = False
    closing = False                 # ESC pressed: saving, then out
    t_shown = time.perf_counter()
    bot_name = sw.name              # what the checkpoint is called; the prompt sets it
    fast = bool(getattr(opts, "swarm_fast", False))   # replay off: breed flat out
    save_mode = str(getattr(opts, "swarm_save", None) or "ask")   # ask | always | never
    discarded = False               # ESC said: no checkpoint
    save_asked = False              # K asked for one: it is kept whatever ESC says
    sw.collect_traces = not fast
    prompt = None                   # the name prompt (garage_ui.TextPrompt), when open
    prompt_then = None              # 'save' (K) or 'close' (ESC): what follows ENTER
    ui_text = None
    try:
        from . import garage_ui as _gui
        ui_text = _gui.Text(_gui.Fonts(1.0))
    except Exception as exc:        # no prompt widget: save under the swarm's name
        print(f"swarm: name prompt unavailable ({type(exc).__name__}: {exc})")

    def _ask_name(then: str):
        """Open the name prompt; ENTER runs `then`. Without the widget, run it now."""
        nonlocal prompt, prompt_then
        if ui_text is None:
            return False
        prompt = _gui.TextPrompt()
        prompt.show("name this bot" if then == "save" else "save the best bot?", "",
                    f"ENTER save (empty = {bot_name})   ESC "
                    + ("discard - no checkpoint" if then == "close" else "cancel"))
        prompt_then = then
        return True

    def _begin_close():
        nonlocal closing, autorun, measuring, note, note_until
        closing = True
        autorun = False
        sw.stop()          # the breeding pool goes; the CPUs are the measurement's
        if sw.best is not None and saved_path is None:
            if measuring is None:
                measuring = sw.measure_best_async()
            note = (f"!saving {bot_name}: measuring its lap at 1 ms ... "
                    "(ESC again = save it now, unmeasured)")
            note_until = time.perf_counter() + 600.0
            return True
        return measuring is not None

    def _save_now(measured=None):
        nonlocal saved_path, saved_measured, note, note_until
        try:
            saved_path, saved_measured = sw.save_best(measured=measured, measure=False,
                                                      name=bot_name)
            lb = (measured or {}).get("best")
            note = (f"!SAVED {saved_path}   "
                    + (f"lap at 1 ms: {lb:.2f} s" if lb else
                       ("lap at 1 ms: no flying lap" if measured else "not re-measured"))
                    + "   drive it: --ml-drive")
            print(note.lstrip("!"), "|", json.dumps(measured or {}, default=str))
        except Exception as exc:
            note = f"!save failed: {type(exc).__name__}: {exc}"
            print(note)
        note_until = time.perf_counter() + 10.0

    try:
        sw.start_evaluation()
        while running:
            dt_wall = (clock.tick(SWARM_FAST_FPS) if fast
                       else clock.tick_busy_loop(opts.fps)) / 1000.0
            now = time.perf_counter()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and prompt is not None:
                    #  the name prompt owns the keyboard while it is open
                    res = prompt.handle(ev)
                    if res is None:
                        continue
                    value = prompt.value
                    prompt = None
                    if res == "ok" and value.strip():
                        bot_name = value.strip()
                    if prompt_then == "save":
                        if res == "ok" and sw.best is not None and measuring is None:
                            save_asked = True
                            measuring = sw.measure_best_async()
                            note, note_until = f"!measuring {bot_name} at 1 ms ...", now + 60.0
                    elif prompt_then == "close":
                        if res == "ok":
                            if not _begin_close():
                                running = False
                        else:              # ESC in the prompt: no checkpoint
                            discarded = True
                            running = False
                    prompt_then = None
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        if closing:
                            #  second ESC: keep the best as it is, no 1 ms lap
                            if measuring is not None:
                                measuring[1].terminate()
                                measuring = None
                            if sw.best is not None and saved_path is None:
                                _save_now()
                            running = False
                        elif sw.best is None or saved_path is not None:
                            running = False
                        elif measuring is not None:
                            #  K already asked for this save: finish it (the
                            #  1 ms lap lands, then out; ESC again = unmeasured)
                            _begin_close()
                        elif save_mode == "never":
                            discarded = True
                            running = False
                        elif save_mode == "ask" and _ask_name("close"):
                            pass           # ENTER / ESC in the prompt continues the close
                        elif not _begin_close():
                            running = False
                    elif k == pygame.K_SPACE:
                        autorun = not autorun
                    elif k == pygame.K_RETURN:
                        want_next = True
                    elif k == pygame.K_k:
                        if sw.best is not None and measuring is None:
                            if not _ask_name("save"):
                                save_asked = True
                                measuring = sw.measure_best_async()
                                note, note_until = "!measuring the best at 1 ms ...", now + 60.0
                    elif k == pygame.K_v:
                        fast = not fast
                        sw.collect_traces = not fast    # from the next generation started
                        note = ("!replay OFF: breeding flat out" if fast
                                else "!replay ON from the next generation scored")
                        note_until = now + 4.0
                    elif k == pygame.K_LEFTBRACKET:
                        speed_i = max(0, speed_i - 1)
                    elif k == pygame.K_RIGHTBRACKET:
                        speed_i = min(len(SWARM_SPEEDS) - 1, speed_i + 1)
                    elif k == pygame.K_c:
                        order = ("car_up", "chase", "world_up")
                        cfgv.mode = order[(order.index(cfgv.mode) + 1) % 3] if cfgv.mode in order else "car_up"
                    elif k == pygame.K_EQUALS:
                        renderer.set_zoom(renderer.zoom_manual * 1.25)
                    elif k == pygame.K_MINUS:
                        renderer.set_zoom(renderer.zoom_manual / 1.25)
                    elif k == pygame.K_0:
                        renderer.set_zoom(1.0)

            # --- the save, when its measurement lands --------------------
            if measuring is not None and measuring[0].ready():
                try:
                    m = measuring[0].get()
                except Exception as exc:
                    m = None
                    print(f"swarm: 1 ms measurement failed: {type(exc).__name__}: {exc}")
                _save_now(m)
                measuring[1].terminate()
                measuring = None
                if closing:
                    running = False

            # --- a scored generation becomes the replay -------------------
            playback_done = (shown is None) or all(
                tau >= r.t_end for r in replays)
            allowed = gens_target <= 0 or (len(sw.history) - gens_done_at_start) < gens_target
            take = (shown is None) or want_next or (autorun and (fast or playback_done))
            if take and not closing and sw.poll():
                want_next = False
                shown = sw.history[-1]
                pop = sw.population
                n = max(len(pop) - 1, 1)
                replays = []
                for rank, ind in enumerate(pop):
                    if not ind.get("trace"):
                        continue           # bred flat out: nothing to replay
                    col = _lerp3(C_SWARM_BEST, C_SWARM_WORST, rank / n)
                    replays.append(_Replay(ind.get("trace") or [], col,
                                           ind.get("ended") == "time",
                                           f"#{ind['id']}"))
                tau = 0.0
                t_shown = time.perf_counter()
                sw.save_state()
                if allowed:
                    sw.reproduce()
                    sw.start_evaluation()
                lb = shown["lap_best"]
                print(f"  gen {shown['gen']:3d}  best {shown['best']:8.1f} m  "
                      f"mean {shown['mean']:8.1f}  lap "
                      + (f"{lb:6.2f} s" if lb else "  --   ")
                      + f"  lapped {shown['n_lapped']}/{sw.pop}  sigma {shown['sigma']:.3f}  "
                      f"{shown['ended']}  [{shown['secs']} s]", flush=True)

            # --- playback ------------------------------------------------
            spd = SWARM_SPEEDS[speed_i]
            if shown is not None and not playback_done:
                tau += dt_wall * spd
            ghosts = []
            hero_st = None
            hud = rnd.HudData()
            if replays:
                for i, r in enumerate(replays):
                    x, y, psi, u, wd, ws, alive, td = r.pose(tau)
                    if i == 0:
                        hero_st = SimpleNamespace(x=x, y=y, psi=psi, u=u, v=0.0)
                        hud.wing_deploy, hud.wing_side = float(wd), int(ws)
                        hud.wing_on = wd > 0.05 or td > 0.05
                        if wd > 0.0 and int(ws) == 0:      # both flanks: the air brake
                            hud.wing_deploy_l = hud.wing_deploy_r = float(wd)
                        hud.top_deploy = float(td)
                        hud.V, hud.V_kmh = u, u * 3.6
                        continue
                    ghosts.append((x, y, psi, r.colour if alive else C_SWARM_DEAD))
            if user_replay is not None:
                x, y, psi, u, wd, ws, alive, _td = user_replay.pose(tau)
                if tau <= user_replay.t_end:
                    ghosts.append((x, y, psi, C_SWARM_USER))
            if hero_st is None:
                x0, y0, psi0 = trk.start_pose(tr)
                hero_st = SimpleNamespace(x=x0, y=y0, psi=psi0, u=0.0, v=0.0)
            hud.ghosts = ghosts
            hc = getattr(opts, "hud_cfg", None)
            if hc:
                for kk, vv in hc.items():          # the garage build's wing geometry
                    setattr(hud, kk, vv)
            hud.x_w = cfg_kwargs.get("x_w", hud.x_w)
            hud.h_w = cfg_kwargs.get("h_w", hud.h_w)
            hud.wing_type = opts.wing

            # --- overlay -------------------------------------------------
            lines = [f"SWARM {sw.name}   {sw.pop} cars   seed {sw.seed_source}   "
                     f"{tr.title or tr.name} / {car_title} / wing {opts.wing}"]
            if shown is not None:
                lb = shown["lap_best"]
                lines.append(f"generation {shown['gen']}:  best {shown['best']:.0f} m   "
                             f"mean {shown['mean']:.0f} m   lapped {shown['n_lapped']}/{sw.pop}   "
                             + (f"best lap {lb:.2f} s" if lb else "no full lap yet"))
                lines.append(f"  {', '.join(f'{k} {v}' for k, v in shown['ended'].items())}"
                             f"   sigma {shown['sigma']:.3f}   computed in {shown['secs']} s")
            if sw.best is not None:
                b = sw.best
                lines.append(f"all-time best: gen {b.get('gen', 0)} car #{b['id']}   "
                             f"{b['reward']:.0f} m   "
                             + (f"lap {b['lap_best']:.2f} s" if b.get("lap_best") else "no lap")
                             + f"   wings {ml_swarm.wings_str(b)}")
            if user_lap:
                lines.append(f"your seed lap: {user_lap:.2f} s  (cyan car)")
            if fast:
                lines.append("!REPLAY OFF - breeding flat out; V to watch the next generation")
                hist = sw.history[-SWARM_TABLE_ROWS:]
                if hist:
                    lines.append("   gen    best m    mean m    lap s   lapped   sigma   secs")
                    for h in hist:
                        lb = h["lap_best"]
                        lines.append(f"  {h['gen']:4d}  {h['best']:8.0f}  {h['mean']:8.0f}  "
                                     + (f"{lb:7.2f}" if lb else "     --")
                                     + f"   {h['n_lapped']:2d}/{sw.pop:<3d}  {h['sigma']:.3f}  "
                                     f"{h['secs']:5.1f}")
            nxt = ("ready" if sw.ready() else "computing ...") if sw._pending is not None else (
                "closing" if closing else ("stopped (--swarm-gens reached)" if not allowed else "-"))
            lines.append(f"next generation: {nxt}   auto-run {'ON' if autorun else 'off'}   "
                         f"t {tau:5.1f} / {sw.T:.0f} s   x{spd:g}")
            lines.append("SPACE auto | ENTER next | K name + save best | V replay | [ ] speed | "
                         "C camera | - = 0 zoom | "
                         + ("ESC back to driving" if embedded else "ESC quit"))
            if saved_path and (not note or now >= note_until):
                lines.append(f"saved: {saved_path}")
            if note and now < note_until:
                lines.append(note)
            hud.overlay = lines

            renderer.update_camera(hero_st, dt_wall)
            renderer.draw_frame(hero_st, None, 0.0, None, hud, None)
            if prompt is not None:
                prompt.draw(renderer.screen, ui_text)
            renderer.present()
            if opts.render == "offscreen" and shown is not None and now - t_shown > 1.5:
                running = False    # offscreen = a smoke test: one generation replayed, then out
    finally:
        try:
            if sw.scored() or sw.history:
                sw.save_state()
                print(f"swarm state: {sw.state_path()}  (--swarm-resume to continue)")
            if (sw.best is not None and saved_path is None and sw.history
                    and (save_asked or (not discarded and save_mode != "never"))):
                #  the window closed some other way: keep the best, unmeasured
                #  (the blocking 1 ms measurement is what froze the window)
                path, _m = sw.save_best(measure=False, name=bot_name)
                opts.swarm_saved = path
                print(f"swarm: saved the best genome (not re-measured) to {path}")
                print(f"  drive it:      python3 -m drive.drive --ml-drive {path}")
                print(f"  breed from it: python3 -m drive.drive --swarm {sw.pop} --swarm-seed {path}")
                print(f"  race it:       python3 -m drive.drive --race {path}"
                      + (f" --race-car {car_name}" if stock else ""))
            elif sw.best is not None and saved_path is None and sw.history:
                print("swarm: the best genome was NOT written as a checkpoint "
                      + ("(ESC discarded it)" if discarded else "(save best: never)")
                      + f"; it is in the state file -- --swarm-resume {sw.state_path()} "
                      "and K to keep it")
            elif saved_path:
                opts.swarm_saved = saved_path
                print(f"  drive it:      python3 -m drive.drive --ml-drive {saved_path}")
                print(f"  breed from it: python3 -m drive.drive --swarm {sw.pop} --swarm-seed {saved_path}")
                print(f"  race it:       python3 -m drive.drive --race {saved_path}"
                      + (f" --race-car {car_name}" if stock else ""))
        finally:
            if measuring is not None:
                measuring[1].terminate()
            sw.close()
            if not embedded:
                pygame.quit()
    return 0


def _resolve_design(opts):
    """The garage's car for this launch -> opts. (garage module, design, lib);
    the module is None when the garage cannot import, and the launch then
    drives the --wing flags as they are."""
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
        opts.garage_lib = lib              # the pre-race page's PICK lists its builds
    except Exception as exc:
        print(f"garage unavailable ({exc})")
        grg = None
    return grg, design, lib


def _player_session(opts) -> bool:
    """A windowed, human-driven session: the only kind that reads or writes
    the per-map builds (a player file). Not `--ml-drive`, not headless, not
    `--render off / offscreen` (the pre-race screen's own rule)."""
    return not (getattr(opts, "ml_drive", None) or getattr(opts, "headless", False)
                or getattr(opts, "render", None) in ("off", "offscreen"))


def _track_build_used(track, design, opts) -> None:
    """This session drives `design` on `track`: it is that map's build from
    now on (the plan's "the last build used on this track")."""
    if not _player_session(opts):
        return
    try:
        from .records import RecordBook
        book = RecordBook()
        js = design.to_json()
        cur = book.last_build(track)
        if not cur or cur.get("build") != js:
            book.set_last_build(track, design.name, js)
    except Exception as exc:               # noqa: BLE001 -- never stops a drive
        print(f"pre-race: per-map build not saved ({type(exc).__name__}: {exc})")


def _autosave_build(design, lib) -> None:
    """Put a car that is in no library file into the library before a PICK
    replaces it (as '<name> (autosave)', '... 2', ...)."""
    try:
        from .prerace import _same_build
        js = design.to_json()
        if any(_same_build(b, js) for b in lib.builds.values()):
            return
        base = f"{design.name} (autosave)"
        name, k = base, 2
        while name in lib.builds:
            name, k = f"{base} {k}", k + 1
        js["name"] = name
        js["builtin"] = False
        lib.save_build(js)
        print(f"pre-race: the car you were driving is saved in the library as '{name}'")
    except Exception as exc:               # noqa: BLE001
        print(f"pre-race: could not autosave the car ({type(exc).__name__}: {exc})")


def _track_build(grg, lib, design, track, opts):
    """The build last USED on `track` (runs/records/last_builds.json), as a
    CarBuild, when it differs from `design`; else None. A player file, so
    only a player session reads it (`_player_session`)."""
    if not _player_session(opts):
        return None
    try:
        from .records import RecordBook
        lb = RecordBook().last_build(track)
        if not lb or lb["build"] == design.to_json():
            return None
        d2 = grg.CarBuild.from_json(lb["build"])
        d2.clamp(lib)
        print(f"pre-race: {track} opens with the build last used there, '{lb.get('name', '')}'")
        return d2
    except Exception as exc:               # noqa: BLE001 -- a bad file: keep the car
        print(f"pre-race: no per-map build ({type(exc).__name__}: {exc})")
        return None


def _challenge_class(settings) -> str:
    """The class key the settings drive (plan D1)."""
    return "|".join((settings.track, settings.car, settings.engine, settings.wet))


def _challenge_restore(opts, settings, keep_changed: str | None = None) -> None:
    """The challenge is over: the player's own track / car / engine / surface
    (and radius / cw) back. `keep_changed` (the challenge's class): a field the
    player changed during it (TAB, the settings page) is kept."""
    prev = getattr(opts, "challenge_prev", None)
    if prev:
        cls = keep_changed.split("|") if keep_changed else None
        for i, k in enumerate(("track", "car", "engine", "wet")):
            if cls is None or getattr(settings, k) == cls[i]:
                setattr(settings, k, prev[k])
        settings.save()
        opts.radius, opts.cw = prev.get("_radius", opts.radius), prev.get("_cw", opts.cw)
    opts.challenge, opts.challenge_prev = None, None


def _challenge_switch(sim, opts, settings) -> None:
    """After a session: a challenge was picked (move the settings to its
    class, remembering the player's own) or ended (put them back). A
    challenge and the driving tutorial shut each other off: a pick ends the
    tutorial (ESC > Tutorial continues it), a tutorial started over a
    challenge ends the challenge."""
    pick = getattr(sim, "challenge_pick", None)
    tut = getattr(opts, "tutorial", None)
    if pick and tut is not None:
        tut.end()
        opts.tutorial = None
        tut = None
    ending = getattr(sim, "challenge_end", False) or (
        tut is not None and not pick and getattr(opts, "challenge", None) is not None)
    if ending and not pick:
        _challenge_restore(opts, settings)
        return
    if not pick:
        return
    from .challenges import load_all
    ch = load_all().get(pick)
    if ch is None:
        return
    if getattr(opts, "challenge_prev", None) is None:
        opts.challenge_prev = dict(track=settings.track, car=settings.car,
                                   engine=settings.engine, wet=settings.wet,
                                   _radius=opts.radius, _cw=opts.cw)
    settings.track, settings.car, settings.engine, settings.wet = ch["class"].split("|")
    settings.save()
    opts.radius, opts.cw = 50.0, False     # the standard skidpad
    opts.challenge = ch
    opts.prerace_skip = True


def _session_car(opts, settings):
    """The track, the car and the VehicleConfig kwargs ONE session drives --
    shared by the interactive session and the swarm, so the swarm's cars are
    exactly the car the user was driving. Returns (tr, car, cfg_kwargs,
    global_wet); `VehicleConfig(**cfg_kwargs)` is the session's config."""
    settings.to_opts(opts)                 # the settings are the truth; opts
    tr = trk.make_track(opts.track, opts.radius, opts.cw,   # is the carrier
                        surfaces=(opts.wet != "none"))
    global_wet = MU_WET_SCALE if opts.wet == "all" else 1.0

    aero_kw = dict(wing=opts.wing, x_w=opts.wing_x, h_w=opts.wing_h,
                   delta_dev_geom=radians(getattr(opts, "wing_inc", 0.0)),
                   dev_flank=getattr(opts, "dev_flank", "outer"))
    if getattr(opts, "wing_cfg", None):
        aero_kw.update(opts.wing_cfg)      # the garage build's three wings
    car = settings.car_spec(getattr(opts, "mass_points", ()) or ())
    cfg_kwargs = dict(aero_kw,
                      abs_on=bool(settings.abs), tc_on=bool(settings.tc),
                      power_scale=settings.power_scale,
                      mu_scale=car.mu_scale)
    return tr, car, cfg_kwargs, global_wet


def _interactive_session(opts, pad=None, garage=False, settings=None):
    """One windowed session. Returns the Sim (stop_reason says why it ended).
    pygame is initialised by the caller and NOT quit here, so the garage and
    the settings page can run several sessions in one window."""
    global _HELP_PRINTED

    if settings is None:
        settings = Settings(path="").apply_cli(opts)
    # The car: the Car setting, plus the ballast, plus the mass of the three
    # wings the garage fitted at their own stations (`drive/aero/wing.
    # wing_mass` -> `CarBuild.mass_points`). With the stock car, no ballast
    # and no wings this is `cars.CORSA_C` itself and `car.mu_scale` is 1.0,
    # so the session is the car every rig measures. `_session_car` is the one
    # place this is assembled, so the swarm breeds exactly this car.
    tr, car, cfg_kwargs, global_wet = _session_car(opts, settings)
    cfg = VehicleConfig(**cfg_kwargs)
    veh = Vehicle(car, cfg)
    x, y, psi = trk.start_pose(tr, 0.0)
    veh.reset(x, y, psi, V=0.0, gear=0 if settings.gearbox == "clutch" else 1)

    # input: keyboard + gamepad. The blend is ALWAYS used (pad may be None):
    # it is what hot-plugs a DualSense that is switched on after launch.
    # The limiter runs on the MEASURED understeer gradient (input.py's own
    # calibration finding): with the spec's 3.2 deg/g the aid capped the
    # driver at 67% of the angle the car needs at R = 50 and the car could
    # never reach its own grip limit from the seat.
    inp = None
    if getattr(opts, "ml_drive", None):
        #  The ML agent in the driver's seat. It is handed the SAME
        #  ScriptedInput closure the acceptance scripts use, so nothing new
        #  reaches the physics path and `drive/ml` stays strictly additive:
        #  no import of it anywhere unless this flag is given.
        inp = _ml_input(opts.ml_drive, tr, opts)
    if inp is None:
        try:
            from . import input as inp_mod
            #  the SELECTED car's road-wheel lock, not the Corsa's on every
            #  car. Exactly inp_mod.DELTA_LOCK_DEG for the Corsa, so the
            #  default session is unchanged.
            from .vehicle import car_lock_rad
            lock_deg = math.degrees(car_lock_rad(car))
            kb = inp_mod.KeyboardInput(steer_limit=settings.steer_aid,
                                       k_us_deg=inp_mod.K_US_DEG_MEASURED,
                                       lock_deg=lock_deg)
            if pad is not None:
                pad.steer_limit = kb.steer_limit
                pad.k_us_deg = kb.k_us_deg
                pad.lock_deg = kb.lock_deg
                pad.delta_deg = 0.0
                pad.seed_edges()           # the button that ended the last
                pad.set_menu(False)        # session is not a press in this one
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
            rnd.set_car(car)               # the HUD's %mg and the g-g envelope
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
    sim.track_radius, sim.track_cw = float(opts.radius), bool(opts.cw)
    #  lap records (drive/records.py): the ONE place runs/records/ is
    #  attached -- scripted and headless runs never read player files
    try:
        from .records import session_recorder
        sim.recorder, why = session_recorder(opts, settings, MU_WET_SCALE, on_lap=sim._rec_lap)
        if why:
            print(f"records: {why}")
    except Exception as exc:               # noqa: BLE001 -- a broken record file
        print(f"records unavailable ({type(exc).__name__}: {exc})")   # never stops a drive
        sim.recorder = None
    #  the pre-race screen (drive/prerace.py): a timed session on a lap map
    #  starts on it; the pause menu's Time trial reaches it any time
    prerace_now = False
    if sim.recorder is not None and renderer is not None:
        try:
            from .prerace import PreRace, wanted
            lib = getattr(opts, "garage_lib", None)
            sim.prerace = PreRace(sim.recorder.key, sim.recorder.book,
                                  getattr(opts, "build_name", "") or "",
                                  getattr(opts, "build_json", None),
                                  builds=dict(getattr(lib, "builds", None) or {}),
                                  titles=dict(track=trk.TRACK_TITLES.get(settings.track,
                                                                         tr.title or tr.name),
                                              car=cars.car_name(settings.car),
                                              engine=engine_label(settings.engine, car),
                                              surface=SURFACE_LABELS[settings.wet]),
                                  can_edit=bool(garage))
            prerace_now = wanted(opts, settings) and not getattr(opts, "prerace_skip", False)
        except Exception as exc:           # noqa: BLE001
            print(f"pre-race screen unavailable ({type(exc).__name__}: {exc})")
            sim.prerace = None
    opts.prerace_skip = False
    #  the driving tutorial (drive/tutorial.py): a player session has the
    #  progress file; a running tutorial comes back on every restart
    sim.progress_file = getattr(opts, "progress", None)
    tut = getattr(opts, "tutorial", None)
    sim.tutorial = tut if (tut is not None and tut.active and sim.progress_file is not None) else None
    sim.tutorial_car = bool(getattr(opts, "tutorial_car", False))
    offer_now = (bool(getattr(opts, "tutorial_offer", False)) and sim.progress_file is not None
                 and renderer is not None and sim.tutorial is None)
    opts.tutorial_offer = False            # once a launch
    if sim.tutorial is not None:
        prerace_now = False                # the tutorial has the car
    #  challenges (drive/challenges.py): a player session with a garage can
    #  list them; a session started for one drives it
    lib = getattr(opts, "garage_lib", None)
    if sim.progress_file is not None and lib is not None:
        sim.challenge_build = (getattr(opts, "build_json", None), lib)
    chal = getattr(opts, "challenge", None)
    if chal is not None and sim.challenge_build is not None:
        try:
            from .challenges import ChallengeRun, build_stats, refusals
            stats = build_stats(sim.challenge_build[0], lib, car, settings.ballast)
            why = refusals(chal["constraints"], stats)
            sim.challenge = ChallengeRun(chal, tr, stats, sim.progress_file)
            prerace_now = False
            if why:                        # listed and endable, never counted
                sim.challenge.refused = why[0]
                print(f"challenge '{chal['title']}' refused: {why[0]}")
                sim._rec_note(f"CHALLENGE refused: {why[0]}", 8.0)
        except Exception as exc:           # noqa: BLE001 -- never stops a drive
            print(f"challenge unavailable ({type(exc).__name__}: {exc})")
    #  the ghosts and the live delta (drive/ghosts.py): with records only
    if sim.recorder is not None:
        try:
            from .ghosts import GhostSet, SLOT_DEFAULT
            sim.ghosts = GhostSet(sim.recorder.book, sim.recorder.key,
                                  slot=getattr(opts, "ghost_slot", None) or SLOT_DEFAULT,
                                  enabled=getattr(opts, "ghosts_on", True))
        except Exception as exc:           # noqa: BLE001 -- a bad trace never stops a drive
            print(f"ghosts unavailable ({type(exc).__name__}: {exc})")
            sim.ghosts = None
    if getattr(opts, "seed_lap", False):
        sim.seed_armed = True              # --seed-lap: K already pressed
        opts.seed_lap = False              # once; a restart is a fresh choice
    if getattr(opts, "swarm_menu", None):
        sim.swarm_opts.update(opts.swarm_menu)
    newest = race_newest_bot()
    if newest:
        sim.race_opts["bot"] = newest      # your newest bot, unless the page says otherwise
    race_menu = getattr(opts, "race_menu", None) or {}
    for k in RACE_MENU_DEFAULTS:
        if race_menu.get(k):
            sim.race_opts[k] = race_menu[k]
    race_spec = getattr(opts, "race", None)
    race_on = False
    if race_spec:
        #  --race BOT[,BOT2[,BOT3]] [--race-car CAR[,CAR2[,CAR3]]]: the grid,
        #  once; the page owns it from here. A car list shorter than the bot
        #  list repeats its last entry.
        specs = [s.strip() for s in str(race_spec).split(",") if s.strip()]
        carl = [c.strip() for c in str(getattr(opts, "race_car", None) or "same").split(",")
                if c.strip()] or ["same"]
        for i in range(1, RACE_GRID_MAX + 1):
            kb, kc = race_slot_keys(i)
            sp = specs[i - 1] if i <= len(specs) else "none"
            sim.race_opts[kb] = race_best_path() if sp == "best" else sp
            sim.race_opts[kc] = carl[i - 1] if i <= len(carl) else carl[-1]
        opts.race = None
        race_on = True
    elif race_menu.get("active"):
        race_on = True                     # a restart keeps the race on
    if race_on:
        sim.start_race()
    sim.hud_cfg = getattr(opts, "hud_cfg", None)   # the build's wing geometry
    if cfg.has_designed():
        sim.wing_on = True                 # a garage build starts armed, as --wing does
    if not _HELP_PRINTED:
        print(KEYS_HELP)
        _HELP_PRINTED = True
    print(f"session: {tr.title or tr.name} | "
          f"car {cars.car_name(settings.car)} {car.m:.0f} kg "
          f"{100 * car.wdist_f:.0f}% front h_cg {car.h_cg:.3f} m"
          + (f" (ballast {settings.ballast_text()})" if settings.ballast > 0 else "")
          + (f" (+{car.m - settings.car_base.m:.1f} kg of wing)"
             if settings.ballast <= 0 and car is not settings.car_base else "")
          + f" | engine {engine_label(settings.engine, car)} | "
          f"gearbox {GEARBOX_LABELS[settings.gearbox]} | "
          f"ABS {'on' if settings.abs else 'off'} | TC {'on' if settings.tc else 'off'} | "
          f"steering aid {'on' if settings.steer_aid else 'off'} | "
          f"surface {SURFACE_LABELS[settings.wet]} | sound {SOUND_LABELS[settings.sound]} | "
          f"wing {opts.wing} x_w {opts.wing_x:+.2f} h_w {opts.wing_h:.2f}")
    if renderer is not None and opts.render != "offscreen" and not opts.headless:
        sim.sound_enabled = True
        sim._audio_apply()
    if offer_now:
        sim.open_tutorial_offer()          # the first launch: WELCOME
    elif prerace_now:
        sim.open_prerace()                 # RACE (ENTER / CROSS) is one press
    try:
        sim.run_interactive()
    finally:
        if telem is not None:
            telem.close()
        if sim.recorder is not None:
            sim.recorder.close()           # the last lap is on disk before the next session reads it
    return sim


if __name__ == "__main__":
    sys.exit(main())
