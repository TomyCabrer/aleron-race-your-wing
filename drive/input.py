"""Driver -> `Controls`: the keyboard/gamepad layer, and nothing else.

Why this is a module of its own, and why it is not a normalised axis
-------------------------------------------------------------------
A keyboard is a switch, and the car is steered by a hand. The naive bridge
between the two is to map the arrow keys onto a [-1, 1] axis, ramp the axis,
and multiply by full lock. That produces a car nobody can drive: at 29 m/s the
whole usable range of the steering (delta_lim = 4.82 deg out of 32.625 deg of
lock) is the first 15% of the axis, so the car goes from straight to spun
inside one tenth of a key press.

What is done here instead is to ramp the ROAD-WHEEL ANGLE ITSELF at a physical
hand rate: 900 deg/s at the steering wheel divided by the 16.0:1 rack gives
56.25 deg/s at the road wheel, so full lock (32.625 deg = 522 deg at the wheel
= half of 2.9 turns lock-to-lock) arrives in 0.58 s. That is a brisk but real
lock-to-lock time for this car, which is what makes the rate physical rather
than a tuning knob. The speed-dependent soft limit delta_lim(V) then caps the
command with a 15% margin over the car's own steady-state cornering angle, so
the driver can still exceed grip and spin it, but cannot ask for 30 deg of lock
at 30 m/s by leaning on a key.

The limiter MUST open up with body slip (+1.2 deg of extra lock per deg of
|beta|). delta_lim(30 m/s) is 4.72 deg and a 12 deg slide needs 10-15 deg of
opposite lock; without the beta term every slide is an unrecoverable spin and
the user concludes the physics is broken when the failure is in the input
layer. Equally, `steer_limit=False` must bypass the ENTIRE aid: every
validation script commands delta directly, and an aid that clipped a commanded
angle would silently contaminate a measured number.

Rate/level separation
---------------------
`poll_events()` runs once per RENDER frame (60 Hz) and is the only thing that
touches the pygame event queue. `update()` runs once per PHYSICS step (1 kHz)
and integrates the ramps at DT_PHYS with the key state held from the last
sample. Integrating the ramps at the frame rate makes the same key press
produce different steering on a 60 fps and a 45 fps machine, and destroys the
determinism of a replayed run.

Shift, wing and camera keys are EDGE-triggered from the event queue only.
Reading `pygame.key.get_pressed()` for E/Q/F inside the physics loop fires a
thousand shift requests per press. The four continuous axes (arrows) and the
three held modifiers (LSHIFT, Z, SPACE) are level-triggered, which is what they
physically are.

Ownership
---------
This module publishes `Controls` from `drive.vehicle` (contract section 4 makes
`vehicle.py` its owner) and fills the fields a human can command. The auto-
clutch and shift state machine that specs/harness.txt eq.7 sketches lives in
`drive/powertrain.py` (`update_shift`), not here: the contract makes `gear_req`
an EDGE consumed by the model, and the wing's 0.45 s deploy lag likewise lives
in `vehicle.py` behind the boolean `wing_on`. This layer emits intent only.

Calibration finding (reported upward, not silently absorbed)
------------------------------------------------------------
harness.txt's own pitfall says to recalibrate K_US_DEG from the model once it
runs. Done: drive.vehicle.steady_state_corner needs 9.569 deg of road wheel to
hold R=50 at the limit, not the spec's assumed 5.614, so the limiter's
understeer gradient is 7.82 deg/g and not 3.2. With the spec's 3.2 the aid caps
the driver 25-42% BELOW the angle the car actually needs (the aid, not grip,
becomes the limit); with 7.82 the designed +15% margin comes back at every
radius. The DEFAULT is left at 3.2 because the contract pins the signature and
V16/V17 are quoted from it; pass `k_us_deg=K_US_DEG_MEASURED` (or run
`python3 -m drive.input --calibrate`) to see the corrected version.

Angles: everything internal to the steering is in DEGREES at the road wheel,
because every number in the spec and every acceptance test is quoted that way.
`Controls.delta` is converted to radians on the way out, once, in `update()`.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Callable, Optional, Protocol, Sequence

# The banner goes to stdout and would land in the middle of a self-check
# report. setdefault, so `PYGAME_HIDE_SUPPORT_PROMPT=` restores it.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402  (must follow the env var above)

from .vehicle import Controls

__all__ = [
    "Controls",
    "InputSource",
    "KeyboardInput",
    "GamepadInput",
    "BlendedInput",
    "ScriptedInput",
    "steer_limit_deg",
    "default_input",
    "self_check",
]


# ---------------------------------------------------------------------------
# Constants. Every one of these is from specs/harness.txt PARAMETERS with its
# provenance; nothing here is free to be re-tuned without moving a spec number.
# ---------------------------------------------------------------------------

G = 9.81                      # m/s^2   as everywhere else in the study
L_WB = 2.491                  # m       corsa_c.L (published)
STEER_RATIO = 16.0            # est +/-1.5   corsa_c; 15.86:1 from 2.9 turns lock-to-lock
DELTA_LOCK_DEG = 32.625       # derived: 2.9 turns = 1044 deg / 2 / 16.0

W_HAND_DRIVE = 900.0          # est, band 600-1200 deg/s at the WHEEL. A normal
                              # driver is 300-500, an emergency input 700-1000;
                              # 900 is the keyboard concession. 522/900 = 0.58 s
                              # to lock, i.e. a real brisk lock-to-lock.
W_HAND_RETURN_LO = 720.0      # est   self-aligning torque grows with speed, so
W_HAND_RETURN_HI = 1440.0     # est   the return rate must too, or the car feels
V_RETURN_FULL = 22.0          # m/s   glued off-centre slow and twitchy fast.

W_DRIVE_DEG = W_HAND_DRIVE / STEER_RATIO          # 56.25 deg/s at the road wheel
W_RETURN_LO_DEG = W_HAND_RETURN_LO / STEER_RATIO  # 45.0
W_RETURN_HI_DEG = W_HAND_RETURN_HI / STEER_RATIO  # 90.0
# Counter-steer = drive + return, i.e. 101.25 -> 146.25 deg/s. Deliberately
# superhuman: without it a keyboard driver cannot catch a slide at all.

AY_MAX_DRY = 8.4608           # m/s^2  qss.max_ay(V) with k=0 (0.8624671 g),
                              # constant in V because the baseline has no
                              # downforce. qss.balance(100): util_f 1.0000.
STEER_MARGIN = 1.15           # derived: the 15% the limiter leaves above the
                              # car's own steady-state angle, so the driver can
                              # still exceed grip. At R=50 needed 5.614 deg vs
                              # limit 6.457 deg.
K_US_DEG = 3.2                # est, band 2.0-5.0 deg road-wheel per g for a FWD
                              # hatch. LIMITER ONLY - never enters the physics.
                              # Kept as the DEFAULT because the contract pins
                              # the signature and V16/V17 are quoted from it.
K_US_DEG_MEASURED = 7.82      # MEASURED, and this is the honest number: the
                              # spec's own pitfall says to recalibrate k_us from
                              # the model once it runs, and
                              # drive.vehicle.steady_state_corner now does:
                              #   R    V      ay      delta   Ackermann  k_us
                              #   30  15.802  0.8484  11.587    4.757    8.05
                              #   50  20.519  0.8584   9.569    2.854    7.82
                              #  100  29.296  0.8749   7.971    1.427    7.48
                              #  130  33.580  0.8842   7.550    1.098    7.30
                              # (`python3 -m drive.input --calibrate` regenerates
                              # this table.) 3.2 is a LINEAR-range gradient
                              # (V_char 25 m/s -> 2.24 deg/g); at the limit the
                              # front axle is at alpha_f = 10.1 deg and the
                              # gradient is 7.3-8.0. Consequence: with k_us=3.2
                              # the aid caps the driver at 6.46 deg where the
                              # car needs 9.57 to hold R=50, i.e. 67% of the
                              # required angle - the aid, not grip, becomes the
                              # limit. With 7.82 the margin comes back to the
                              # designed +15% (15.0/15.5/17.6% at R=30/50/100).
                              # Pass k_us_deg=K_US_DEG_MEASURED to drive it.
BETA_LOCK_GAIN = 1.2          # est, band 0.8-1.5, deg of extra lock per deg of
                              # body slip. 4.72 + 1.2*12 = 19.1 deg of opposite
                              # lock available in a 12 deg slide.
V_STEER_FLOOR = 3.0           # m/s  the max(V, 3) in delta_ss; below it the
                              # 1/V^2 blows up and the lock clamp takes over.

# Pedal ramp rates, 1/s (rise, fall). All estimated; a real driver's foot is
# 0.15-0.3 s across the pedal and published emergency brake application times
# are 0.15-0.25 s. Clutch is asymmetric because the RELEASE rate is what makes
# or breaks a launch.
RATE_THROTTLE = (3.5, 6.0)    # est   0->1 in 0.2857 s, 1->0 in 0.1667 s
RATE_BRAKE = (5.0, 8.0)       # est   0->1 in 0.2000 s, 1->0 in 0.1250 s
RATE_CLUTCH = (8.0, 3.0)      # est   press 0.125 s, release 0.333 s
RATE_HANDBRAKE = (8.0, 10.0)  # est   cable lever
FINE_PEDAL_CAP = 0.50         # LSHIFT: half rates on the steer, 50% pedal travel

KEY_SAMPLE_HZ = 60.0          # documentation only: poll_events() is called at
                              # the render rate, which is all pygame offers.

# Gamepad. est; SDL2 Xbox/DualShock layout on macOS. UNVERIFIED against real
# hardware: pygame.joystick.get_count() == 0 on this machine, so the keyboard
# path below is the default and the only one that has ever run.
PAD_DEADZONE_STICK = 0.12
PAD_DEADZONE_TRIG = 0.05
PAD_EXPO = 1.5
PAD_RATE_DEG = 250.0          # deg/s road wheel; a rate cap, not a ramp - the
                              # stick is an absolute axis and already has a
                              # position, so ramping it would double-lag it.
PAD_BLEND_EPS = 0.02          # |pad| above this wins the axis from the keyboard

DEFAULT_PAD_BUTTONS = {
    0: "handbrake",
    1: "wing",
    2: "clutch",
    3: "reset",
    4: "shift_down",
    5: "shift_up",
    6: "menu",           # start     pause menu (controls + reset)
    7: "full_reset",
}
GENERIC_BUTTON_NAMES = {0: "a", 1: "b", 2: "x", 3: "y", 4: "back", 5: "guide",
                        6: "start", 7: "l3", 8: "r3", 9: "l1", 10: "r1",
                        11: "up", 12: "down", 13: "left", 14: "right"}
GENERIC_AXES = {"steer": 0, "ly": 1, "rx": 2, "ry": 3}   # + throttle/brake by count

# PlayStation layout: DualSense (PS5) and DualShock 4 through SDL2's HIDAPI
# driver, which is what pygame 2.5 / SDL 2.28 uses for a Bluetooth pad on
# macOS.  Axes 0 LX, 1 LY, 2 RX, 3 RY, 4 L2, 5 R2 (triggers REST AT -1).
# Buttons 0 cross, 1 circle, 2 square, 3 triangle, 4 create, 5 PS, 6 options,
# 7 L3, 8 R3, 9 L1, 10 R1, 11-14 d-pad up/down/left/right, 15 touchpad.
# Verify with `python3 -m drive.drive --pad-calib`; if the OS hands SDL a
# different ordering, drop the corrected indices in ~/.carsim_pad.json:
#   {"steer": 0, "throttle": 5, "brake": 4, "buttons": {"10": "shift_up"}}
PS_AXES = {"steer": 0, "ly": 1, "rx": 2, "ry": 3, "brake": 4, "throttle": 5}
PS_BUTTON_NAMES = {0: "cross", 1: "circle", 2: "square", 3: "triangle",
                   4: "create", 5: "ps", 6: "options", 7: "l3", 8: "r3",
                   9: "l1", 10: "r1", 11: "up", 12: "down", 13: "left",
                   14: "right", 15: "touchpad"}
PS_PAD_BUTTONS = {
    0: "handbrake",      # cross     held
    1: "wing",           # circle    flank-wing toggle
    2: "clutch",         # square    held
    3: "wing_side",      # triangle  auto / left / right
    4: "reset",          # create    back to the last sector line
    6: "menu",           # options   pause menu (controls + reset)
    7: "zoom_auto",      # L3
    8: "camera",         # R3
    9: "shift_down",     # L1
    10: "shift_up",      # R1
    11: "hud",           # d-pad up
    12: "vectors",       # d-pad down
    13: "slowmo",        # d-pad left
    14: "normal_speed",  # d-pad right
    15: "garage",        # touchpad  back to the 3D editor
}
PAD_CONFIG_PATH = os.path.expanduser("~/.carsim_pad.json")
PAD_RUMBLE = os.environ.get("CARSIM_NO_RUMBLE", "") == ""
PAD_RUMBLE_HZ = 20.0

# Gearbox modes (drive.py's settings; powertrain.update_shift's docstring).
#   name      auto_gearbox  auto_clutch   what the driver does
#   auto      True          True          steer and pedal
#   manual    False         True          + shifts (E/Q, R1/L1); never stalls
#   clutch    False         False         + launches on the clutch (Z / SQUARE),
#                                           can stall; clutch fully in restarts
GEARBOX_MODES = ("auto", "manual", "clutch")
GEARBOX_LABELS = {"auto": "Automatic", "manual": "Manual (auto clutch)",
                  "clutch": "Manual + clutch pedal"}
GEARBOX_HUD = {"auto": "AUTO", "manual": "MAN", "clutch": "MAN+CL"}


def gearbox_flags(mode: str) -> tuple[bool, bool]:
    """(auto_gearbox, auto_clutch) for a mode name; unknown -> automatic."""
    if mode == "manual":
        return False, True
    if mode == "clutch":
        return False, False
    return True, True

# Pause menu (drive/menu.py). While it is open the inputs stop emitting
# driving edges and emit these instead. Names are the layout's button names.
MENU_PAD_NAMES = {
    "ps": {"up": "nav_up", "down": "nav_down", "cross": "select",
           "circle": "back", "options": "menu", "create": "reset",
           "touchpad": "garage"},
    "generic": {"up": "nav_up", "down": "nav_down", "a": "select", "b": "back",
                "start": "menu", "back": "reset"},
}


def detect_pad_layout(name: str) -> str:
    """'ps' for a Sony pad, 'generic' (SDL Xbox order) for anything else."""
    n = (name or "").lower()
    if any(k in n for k in ("dualsense", "dualshock", "ps5", "ps4", "sony",
                            "wireless controller")):
        return "ps"
    return "generic"


def load_pad_config(path: str = PAD_CONFIG_PATH) -> dict:
    """~/.carsim_pad.json -> mapping overrides. Missing file -> {}."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for k in ("steer", "throttle", "brake", "ly", "rx", "ry"):
        if k in raw:
            out[k] = int(raw[k])
    if isinstance(raw.get("buttons"), dict):
        out["buttons"] = {int(b): str(c) for b, c in raw["buttons"].items()}
    return out


# ---------------------------------------------------------------------------
# Key bindings. Single source of truth; drive.py prints this for --help and H.
# ---------------------------------------------------------------------------

# Level-triggered (the four continuous axes plus three held modifiers). These
# are what they physically are: a pedal is held down, not tapped.
HELD_KEYS = {
    "up": pygame.K_UP,
    "down": pygame.K_DOWN,
    "left": pygame.K_LEFT,
    "right": pygame.K_RIGHT,
    "clutch": pygame.K_z,
    "handbrake": pygame.K_SPACE,
    "starter": pygame.K_s,
}

# Edge-triggered, KEYDOWN only. A thousand physics steps per press otherwise.
EDGE_KEYS = {
    pygame.K_e: "shift_up",
    pygame.K_q: "shift_down",
    pygame.K_f: "wing",
    pygame.K_g: "wing_side",
    pygame.K_r: "reset",              # + SHIFT -> 'full_reset'
    pygame.K_p: "pause",
    pygame.K_o: "step",
    pygame.K_LEFTBRACKET: "slowmo",
    pygame.K_RIGHTBRACKET: "normal_speed",
    pygame.K_c: "camera",
    pygame.K_MINUS: "zoom_out",
    pygame.K_EQUALS: "zoom_in",
    pygame.K_0: "zoom_auto",
    pygame.K_h: "hud",
    pygame.K_v: "vectors",
    pygame.K_b: "gg",
    pygame.K_n: "skid",
    pygame.K_x: "clear_skid",       # name matches drive.py's handle_event
    pygame.K_t: "wet",
    pygame.K_m: "marker",
    pygame.K_l: "record",
    pygame.K_TAB: "track_next",
    pygame.K_BACKSPACE: "garage",   # back to the 3D editor (drive --garage)
    pygame.K_ESCAPE: "menu",        # pause menu: controls + reset / quit
}

# Keys while the pause menu is open. Nothing else gets through (a shift or a
# wing toggle from a stray key while reading the controls would be a surprise).
MENU_KEYS = {
    pygame.K_UP: "nav_up",
    pygame.K_DOWN: "nav_down",
    pygame.K_RETURN: "select",
    pygame.K_KP_ENTER: "select",
    pygame.K_SPACE: "select",
    pygame.K_ESCAPE: "menu",
    pygame.K_p: "menu",
    pygame.K_r: "reset",              # + SHIFT -> 'full_reset'
    pygame.K_BACKSPACE: "garage",
}

# The same bindings as rows for the on-screen menu (drive/menu.py).
MENU_HELP_KB = [
    ("UP / DOWN", "throttle / brake"),
    ("LEFT / RIGHT", "steer"),
    ("LSHIFT", "fine: half rates, 50% pedal"),
    ("Z / SPACE", "clutch / handbrake (hold)"),
    ("S", "starter"),
    ("E / Q", "shift up / down"),
    ("F / G", "flank wing / wing side"),
    ("R / SHIFT+R", "reset to sector / full reset"),
    ("P / O", "pause / single step"),
    ("[ / ]", "slow-mo 0.25x / 1x"),
    ("C", "camera"),
    ("- / = / 0", "zoom out / in / auto"),
    ("H / V / B", "HUD / vectors / g-g"),
    ("N / X", "skid marks / clear"),
    ("T", "wet toggle"),
    ("M / L", "telemetry marker / record"),
    ("TAB", "next map"),
    ("BACKSPACE", "garage (3D panel editor)"),
    ("ESC", "this menu / settings"),
]
MENU_HELP_PAD = {
    "ps": [
        ("R2 / L2", "throttle / brake"),
        ("left stick", "steer"),
        ("R1 / L1", "shift up / down"),
        ("CROSS / SQUARE", "handbrake / clutch (hold)"),
        ("CIRCLE", "flank wing"),
        ("TRIANGLE", "wing side"),
        ("CREATE", "reset to sector"),
        ("d-pad UP/DOWN", "HUD / vectors"),
        ("d-pad L/R", "slow-mo / normal"),
        ("R3 / L3", "camera / auto zoom"),
        ("touchpad", "garage (3D panel editor)"),
        ("OPTIONS", "this menu / settings"),
    ],
    "generic": [
        ("RT / LT", "throttle / brake"),
        ("left stick", "steer"),
        ("RB / LB", "shift up / down"),
        ("A / X", "handbrake / clutch (hold)"),
        ("B", "flank wing"),
        ("Y", "reset to sector"),
        ("BACK / START", "full reset / this menu"),
    ],
}
MENU_NO_PAD = "no controller: pair the DualSense (CREATE+PS) - it hot-plugs"


def menu_help(layout: str | None) -> list:
    """Sections for drive/menu.py: keyboard, plus the attached pad's layout."""
    secs = [("KEYBOARD", MENU_HELP_KB)]
    if layout in MENU_HELP_PAD:
        secs.append(("PS5 DUALSENSE" if layout == "ps" else "GAMEPAD",
                     MENU_HELP_PAD[layout]))
    return secs

KEY_HELP = """\
ARROW UP throttle | ARROW DOWN brake | ARROW LEFT/RIGHT steer | LSHIFT fine (half rates, 50% pedal)
Z clutch | SPACE handbrake | S starter | E shift up | Q shift down
F flank-wing toggle | G cycle wing side (auto / left / right)
R reset to last sector line | SHIFT+R full reset (clears skid marks and timing)
P pause | O single physics step while paused | [ ] slow-mo 0.25x / 1.0x
C camera cycle | - / = zoom | 0 auto zoom | H HUD cycle | V force vectors | B g-g | N skid | X clear skid
T toggle wet (global mu_scale 1.0 <-> 0.632) | M telemetry marker | L toggle recording
TAB next track | BACKSPACE garage (3D panel editor) | ESC menu (controls, reset, quit)
PS5 pad: R2 throttle | L2 brake | L-stick steer | R1/L1 shift | CROSS handbrake | SQUARE clutch
         CIRCLE wing | TRIANGLE wing side | OPTIONS menu | CREATE reset | TOUCHPAD garage
         d-pad: up HUD, down vectors, left slow-mo, right normal | R3 camera | L3 auto zoom"""


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def steer_limit_deg(V: float, beta_deg: float, ay_max: float = AY_MAX_DRY,
                    L: float = L_WB, k_us_deg: float = K_US_DEG,
                    lock_deg: float = DELTA_LOCK_DEG,
                    beta_gain: float = BETA_LOCK_GAIN) -> float:
    """The speed-dependent soft lock, in deg at the road wheel (harness eq.5).

    delta_ss is the car's own steady-state cornering equation - Ackermann
    L/R plus an understeer gradient - evaluated at a_ref = 1.15*ay_max, so the
    limit sits exactly 15% ABOVE what the car needs at the grip limit. That
    margin is deliberate: the driver must be able to over-slow the front axle
    and spin it, or the sim is a rail.

    The `beta_gain*|beta_deg|` term is what makes a slide catchable, and the
    outer min() is why it can never ask for more than mechanical lock.

    NOTE this function is the SYMMETRIC form and it stays that way: the
    contract pins the signature and V16/V17 are quoted from it. What the input
    layer actually clamps against is `steer_limit_pair_deg` below, which spends
    the beta bonus on the counter-steer side only. See that docstring for why.

    Pure; unit-tested against the whole delta_lim(V) table in self_check().
    """
    a_ref = STEER_MARGIN * ay_max
    delta_ss = math.degrees(L * a_ref / max(V, V_STEER_FLOOR) ** 2) + k_us_deg * (a_ref / G)
    delta_lim = min(lock_deg, delta_ss)
    return min(lock_deg, delta_lim + beta_gain * abs(beta_deg))


def steer_limit_pair_deg(V: float, beta_deg: float, ay_max: float = AY_MAX_DRY,
                         L: float = L_WB, k_us_deg: float = K_US_DEG,
                         lock_deg: float = DELTA_LOCK_DEG,
                         beta_gain: float = BETA_LOCK_GAIN) -> tuple[float, float]:
    """(limit on LEFT lock, limit on RIGHT lock), both positive magnitudes, deg.

    Same floor as `steer_limit_deg`; the difference is WHERE the beta bonus
    goes. The bonus exists to make a slide catchable (contract section 6), and
    catching a slide means COUNTER-steer. Spending it on both sides makes the
    aid a positive feedback loop instead: more lock -> more slide -> more |beta|
    -> a higher limit -> more lock. Measured on the reported bug, full brake
    from 30 m/s with the aid at k_us_deg = 7.82:

        beta       limit (old, symmetric)      yaw rate delivered
        0 deg          9.30 deg                 0.480 rad/s  (near the peak)
       20 deg         32.625 deg  (full lock)   0.440 rad/s at 36 deg alpha_f

    i.e. the aid handed the driver 23 deg of extra lock that BOUGHT NOTHING --
    the front axle was already 26 deg past the MF peak (10.35 deg) and yaw
    response falls monotonically beyond it. From the seat that is exactly
    "I cannot steer while braking": full lock on, car going straight.

    Sign: `beta = atan2(v, |u|)` with `v` the LEFTWARD velocity, so in a left
    turn at the limit beta is NEGATIVE (contract section 9 item 7 states the
    same thing) and the catch is RIGHT lock. Counter-steer therefore has the
    SAME sign as beta, and the bonus goes to the `sign(beta_deg)` side. The
    magnitude of the catch is untouched: a 12 deg slide still opens 19.12 deg
    of opposite lock at 30 m/s, which is the V17 number.

    Pure.
    """
    a_ref = STEER_MARGIN * ay_max
    delta_ss = math.degrees(L * a_ref / max(V, V_STEER_FLOOR) ** 2) + k_us_deg * (a_ref / G)
    floor = min(lock_deg, delta_ss)
    bonus = min(lock_deg, floor + beta_gain * abs(beta_deg))
    if beta_deg > 0.0:
        return bonus, floor            # slide to be caught with LEFT lock
    if beta_deg < 0.0:
        return floor, bonus            # ... with RIGHT lock
    return floor, floor


def _return_rate_deg(V: float) -> float:
    """Road-wheel self-centring rate, 45 deg/s at rest -> 90 deg/s above 22 m/s."""
    f = _clamp(V / V_RETURN_FULL, 0.0, 1.0)
    return (W_HAND_RETURN_LO + (W_HAND_RETURN_HI - W_HAND_RETURN_LO) * f) / STEER_RATIO


def _ramp(x: float, tgt: float, rise: float, fall: float, dt: float) -> float:
    """First-order-hold pedal ramp; harness eq.3. Never overshoots the target."""
    if tgt > x:
        return min(tgt, x + rise * dt)
    return max(tgt, x - fall * dt)


def _deadzone(a: float, z: float) -> float:
    """Rescaled deadzone: the output still reaches +/-1 at the axis stop."""
    m = abs(a)
    if m < z:
        return 0.0
    return math.copysign((m - z) / (1.0 - z), a)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class InputSource(Protocol):
    """The only thing drive.py knows about a driver.

    Two rates, deliberately: poll_events() at the RENDER rate (it drains the
    pygame queue and re-samples held keys), update() at the PHYSICS rate (it
    integrates the ramps at dt with the state held from the last sample).
    """

    def poll_events(self) -> list[str]:
        """Drain the event queue; return discrete commands ('reset', 'pause', ...)."""
        ...

    def update(self, dt: float, V: float, beta_deg: float,
               rpm: float, gear: int) -> Controls:
        """One physics step of driver state. Returns the command to the model."""
        ...


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


class KeyboardInput:
    """Arrow keys through a hand-rate steering ramp and four pedal ramps.

    The steering state `delta_deg` is the ROAD-WHEEL angle, positive LEFT
    (contract section 0). specs/harness.txt eq.4 writes `d_in = +1 if
    KEY_RIGHT` and then annotates "RIGHT is -delta by SAE sign"; the contract's
    sign convention wins, so the direction input is formed as
    (left - right) and holding RIGHT drives delta negative. Every acceptance
    number is quoted as a magnitude, so nothing else moves.
    """

    def __init__(self, steer_limit: bool = True, fine_key: int = pygame.K_LSHIFT,
                 auto_gearbox: bool = True, wing_on: bool = False,
                 held_keys: dict | None = None, k_us_deg: float = K_US_DEG):
        self.steer_limit = bool(steer_limit)
        self.fine_key = int(fine_key)
        self.auto_gearbox = bool(auto_gearbox)
        self.auto_clutch = True
        self.k_us_deg = float(k_us_deg)   # limiter only; see K_US_DEG_MEASURED
        self.key_map = dict(HELD_KEYS if held_keys is None else held_keys)

        # Continuous state, all integrated in update() at DT_PHYS.
        self.delta_deg = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.handbrake = 0.0

        # Latched driver state.
        self.wing_on = bool(wing_on)
        self.wing_side_mode = "auto"          # 'auto' | 'left' | 'right'; HUD only
        self.paused = False
        self.menu = False                     # pause menu open: nav keys only

        # Level state, re-sampled every poll_events() and HELD between polls.
        self.held = {k: False for k in self.key_map}
        self.held["fine"] = False
        self._override: dict | None = None    # set by set_keys() for scripts/tests

        # Edge state, consumed once by update().
        self._pending_gear = 0
        self._pending_starter = False

        # Diagnostics for the HUD / telemetry.
        self.delta_lim_deg = DELTA_LOCK_DEG
        self.n_events = 0
        # The eased soft-lock bounds, one per direction (see update()). They
        # start wide open so the first step cannot clamp a car that is already
        # turned, and they only ever fall at the self-centring rate.
        self._lim_l = DELTA_LOCK_DEG
        self._lim_r = DELTA_LOCK_DEG

    # -- key state ---------------------------------------------------------

    def set_keys(self, **flags) -> None:
        """Hold a synthetic key state (scripted runs, tests, replays).

        Once called, `poll_events()` stops overwriting the held state from the
        real keyboard - so a test can drive the exact same ramp code the human
        path uses without a window or a physical key. `set_keys(clear=True)`
        hands control back to pygame.
        """
        if flags.pop("clear", False):
            self._override = None
            return
        ov = dict(self._override or {})
        for k, v in flags.items():
            if k not in self.held:
                raise KeyError(f"unknown held key {k!r}; known: {sorted(self.held)}")
            ov[k] = bool(v)
        self._override = ov
        self.held.update(ov)

    def _sample_keys(self) -> None:
        if self._override is not None:
            self.held.update(self._override)
            return
        try:
            pressed = pygame.key.get_pressed()
        except pygame.error:
            return                      # no video system yet; keep the last sample
        for name, code in self.key_map.items():
            self.held[name] = bool(pressed[code])
        self.held["fine"] = bool(pressed[self.fine_key])

    # -- events ------------------------------------------------------------

    def poll_events(self) -> list[str]:
        """Once per RENDER frame. Edges from the queue, levels re-sampled.

        Returns every command string, including 'shift_up' / 'shift_down' /
        'wing' / 'wing_side', which this object has ALREADY folded into its own
        state. drive.py must treat those three as notifications (HUD, telemetry
        marks) and must not apply them a second time.
        """
        cmds: list[str] = []
        try:
            events = pygame.event.get()
        except pygame.error:
            events = []                 # display not initialised: scripted path
        for ev in events:
            if ev.type == pygame.QUIT:
                cmds.append("quit")
                continue
            if ev.type != pygame.KEYDOWN:
                continue
            self.n_events += 1
            cmd = (MENU_KEYS if self.menu else EDGE_KEYS).get(ev.key)
            if cmd is None:
                continue
            if cmd == "reset" and (ev.mod & pygame.KMOD_SHIFT):
                cmd = "full_reset"
            cmds.append(cmd)
            if not self.menu:
                self._apply_command(cmd)
        if self.menu:
            # pedals off while the menu is up; the real keys are re-sampled
            # on the first poll after it closes
            for k in self.held:
                self.held[k] = False
        else:
            self._sample_keys()
        return cmds

    def set_menu(self, flag: bool) -> None:
        self.menu = bool(flag)

    def set_gearbox(self, mode: str) -> None:
        """'auto' | 'manual' | 'clutch' -> the Controls flags (GEARBOX_MODES)."""
        self.auto_gearbox, self.auto_clutch = gearbox_flags(mode)

    def _apply_command(self, cmd: str) -> None:
        """The edges this layer owns. Everything else belongs to drive.py."""
        if cmd == "shift_up":
            self._pending_gear = +1     # queue depth 1: a second press before
        elif cmd == "shift_down":       # update() consumes it overwrites, it
            self._pending_gear = -1     # does not stack into a double shift
        elif cmd == "wing":
            self.wing_on = not self.wing_on
        elif cmd == "wing_side":
            order = ("auto", "left", "right")
            self.wing_side_mode = order[(order.index(self.wing_side_mode) + 1) % 3]
        elif cmd == "pause":
            self.paused = not self.paused

    # -- per physics step --------------------------------------------------

    def update(self, dt: float, V: float = 0.0, beta_deg: float = 0.0,
               rpm: float = 0.0, gear: int = 1) -> Controls:
        """One physics step. Integrates every ramp at dt with the held state."""
        h = self.held
        fine = h.get("fine", False)

        # --- pedals (eq.3) -------------------------------------------------
        self.throttle = _ramp(self.throttle, 1.0 if h.get("up") else 0.0,
                              *RATE_THROTTLE, dt)
        self.brake = _ramp(self.brake, 1.0 if h.get("down") else 0.0,
                           *RATE_BRAKE, dt)
        self.clutch = _ramp(self.clutch, 1.0 if h.get("clutch") else 0.0,
                            *RATE_CLUTCH, dt)
        self.handbrake = _ramp(self.handbrake, 1.0 if h.get("handbrake") else 0.0,
                               *RATE_HANDBRAKE, dt)
        if fine:
            # Clamp the STATE, not just the output: capping only the returned
            # value would make the pedal jump the instant LSHIFT is released.
            self.throttle = min(self.throttle, FINE_PEDAL_CAP)
            self.brake = min(self.brake, FINE_PEDAL_CAP)

        # --- steering (eq.4): ramp the ANGLE, never a normalised axis ------
        d_in = (1 if h.get("left") else 0) - (1 if h.get("right") else 0)
        w_drv = W_DRIVE_DEG
        w_ret = _return_rate_deg(V)
        w_cnt = w_drv + w_ret
        if fine:
            w_drv *= 0.5
            w_ret *= 0.5
            w_cnt *= 0.5

        d = self.delta_deg
        if d_in == 0:
            step = w_ret * dt
            d = 0.0 if abs(d) <= step else d - math.copysign(step, d)
        elif d_in * d < 0.0:
            d += d_in * w_cnt * dt      # counter-steer / flick back
        else:
            d += d_in * w_drv * dt
        self.delta_deg = d

        # --- limiter (eq.5). steer_limit=False bypasses the aid entirely ---
        # Two changes against the original symmetric hard clamp, both from the
        # reported bug (.handoff/08-steering.md):
        #   * the bound is a PAIR and the beta bonus goes to the counter-steer
        #     side only -- see steer_limit_pair_deg;
        #   * the bound is eased DOWN at the self-centring rate instead of
        #     snapping. Opening up is instant (a wider limit is never a
        #     surprise), closing is not: with a directional bonus the bound on
        #     one side can fall by 20+ deg the instant beta changes sign, and a
        #     hard clamp would then teleport the road wheel. Easing it is also
        #     what the wheel physically does - the self-aligning torque pulls
        #     it back at exactly this rate.
        if self.steer_limit:
            lim_l, lim_r = steer_limit_pair_deg(V, beta_deg,
                                                k_us_deg=self.k_us_deg)
        else:
            lim_l = lim_r = DELTA_LOCK_DEG
        ease = w_ret * dt
        self._lim_l = lim_l if lim_l >= self._lim_l else max(lim_l, self._lim_l - ease)
        self._lim_r = lim_r if lim_r >= self._lim_r else max(lim_r, self._lim_r - ease)
        self.delta_deg = _clamp(self.delta_deg, -self._lim_r, self._lim_l)
        # The HUD reads ONE number: report the bound that is actually binding
        # the direction the wheel is turned (at centre they are equal anyway).
        self.delta_lim_deg = self._lim_r if self.delta_deg < 0.0 else self._lim_l

        gear_req, self._pending_gear = self._pending_gear, 0
        starter = bool(h.get("starter")) or self._pending_starter
        self._pending_starter = False

        return Controls(
            delta=math.radians(self.delta_deg),
            throttle=self.throttle,
            brake=self.brake,
            clutch=self.clutch,
            handbrake=self.handbrake,
            gear_req=gear_req,
            auto_gearbox=self.auto_gearbox,
            auto_clutch=self.auto_clutch,
            wing_on=self.wing_on,
            starter=starter,
        )

    def reset(self) -> None:
        """Drop every ramp to zero. Called by drive.py on R / SHIFT+R."""
        self.delta_deg = 0.0
        self.throttle = self.brake = self.clutch = self.handbrake = 0.0
        self._pending_gear = 0
        self._pending_starter = False
        self._lim_l = self._lim_r = DELTA_LOCK_DEG
        self.delta_lim_deg = DELTA_LOCK_DEG


# ---------------------------------------------------------------------------
# Gamepad
# ---------------------------------------------------------------------------


class GamepadInput:
    """Absolute axes: the ramps are bypassed and only a rate cap is applied.

    A stick already HAS a position, so ramping it towards a target would add a
    second lag on top of the driver's hand and make the pad feel worse than the
    keyboard. What is kept is PAD_RATE = 250 deg/s at the road wheel, which is
    fast enough to be invisible to a human and slow enough to stop a dropped
    USB frame from stepping the steering.

    Two layouts. A Sony pad (DualSense / DualShock 4, detected by name) gets
    PS_AXES / PS_PAD_BUTTONS: R2 throttle, L2 brake, R1/L1 shift, cross
    handbrake. Anything else gets the SDL2 Xbox ordering that was here before.
    `~/.carsim_pad.json` overrides either, `mapping=` overrides both.

    The Sony map was written from SDL's HIDAPI PS5 driver ordering, not
    measured on this machine (the pads were paired but asleep). Two guards:
    `--pad-calib` prints live axes/buttons with their names, and a trigger
    axis that does not REST at -1 prints a one-line warning at attach time,
    because that is the signature of the OS handing SDL a different order.

    `joystick=` accepts any object with get_numaxes/get_axis/get_numbuttons/
    get_button/get_name, which is how the axis maths is unit-tested without
    a device.
    """

    def __init__(self, index: int = 0, mapping: dict | None = None,
                 steer_limit: bool = True, joystick=None,
                 k_us_deg: float = K_US_DEG, layout: str | None = None,
                 user_config: bool = True):
        if joystick is None:
            if not GamepadInput.available():
                raise RuntimeError("no gamepad: pygame.joystick.get_count() == 0")
            pygame.joystick.init()
            joystick = pygame.joystick.Joystick(index)
            try:
                joystick.init()
            except (AttributeError, pygame.error):
                pass                    # pygame 2 auto-inits; older ones do not
        self.joy = joystick
        self.index = index
        self.steer_limit = bool(steer_limit)
        self.k_us_deg = float(k_us_deg)

        n = self.joy.get_numaxes()
        self.layout = layout or detect_pad_layout(self.name)
        if self.layout == "ps":
            self.map = dict(PS_AXES)
            self.map["buttons"] = dict(PS_PAD_BUTTONS)
            self.names = dict(PS_BUTTON_NAMES)
        else:
            # specs/harness.txt's fallback branch reads "elif >= 3: axis2 =
            # throttle, axis5 = brake", which cannot be right - axis5 needs 6
            # axes. Taken as a typo and read as the 3-axis SDL layout (2 =
            # combined triggers, 3 = brake where present).
            if n >= 6:
                axis_thr, axis_brk = 5, 4
            elif n >= 4:
                axis_thr, axis_brk = 2, 3
            elif n >= 3:
                axis_thr, axis_brk = 2, -1
            else:
                axis_thr, axis_brk = -1, -1
            self.map = dict(GENERIC_AXES)
            self.map.update({"throttle": axis_thr, "brake": axis_brk,
                             "buttons": dict(DEFAULT_PAD_BUTTONS)})
            self.names = dict(GENERIC_BUTTON_NAMES)
        if user_config:
            user = load_pad_config()
            if user:
                if "buttons" in user:
                    self.map["buttons"].update(user.pop("buttons"))
                self.map.update(user)
        if mapping:
            mapping = dict(mapping)
            if "buttons" in mapping:
                self.map["buttons"] = dict(mapping.pop("buttons"))
            self.map.update(mapping)
        self._name_to_btn = {v: k for k, v in self.names.items()}
        self._rumble_ok: Optional[bool] = None
        self._rumble_t = 0.0
        self._rumble_last = (0.0, 0.0)
        self._rest_checked = False
        self._trig_rest = {"throttle": -1.0, "brake": -1.0}
        self._check_rest()

        # the eased directional soft-lock bounds, as on the keyboard
        self._lim_l = DELTA_LOCK_DEG
        self._lim_r = DELTA_LOCK_DEG
        self.delta_deg = 0.0
        self.axis_steer = 0.0           # post-deadzone stick, for the blend rule
        self.throttle = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.handbrake = 0.0
        self.wing_on = False
        self.wing_side_mode = "auto"
        self.auto_gearbox = True
        self.auto_clutch = True
        self.delta_lim_deg = DELTA_LOCK_DEG
        self._pending_gear = 0
        self._buttons_prev: dict[int, bool] = {}
        self.menu = False
        self._menu_prev: dict[str, bool] = {}
        self._menu_stick = None

    def set_gearbox(self, mode: str) -> None:
        self.auto_gearbox, self.auto_clutch = gearbox_flags(mode)

    def seed_edges(self) -> None:
        """Take the live button state as 'already pressed', so a button held
        across a session boundary (CROSS that selected 'Drive' in the garage,
        or 'Back to the garage' in the drive's menu) is not a fresh edge in
        the session that follows."""
        for btn in self.map["buttons"]:
            self._buttons_prev[btn] = self._button(btn)

    def set_menu(self, flag: bool) -> None:
        """Enter / leave menu mode. Entering SEEDS the edge state from the
        live buttons, so the OPTIONS press that opened the menu does not
        read as a second edge and close it again on the next frame."""
        flag = bool(flag)
        if flag and not self.menu:
            table = MENU_PAD_NAMES.get(self.layout, MENU_PAD_NAMES["generic"])
            self._menu_prev = {n: self.pressed(n) for n in table}
            from .menu import StickNav
            self._menu_stick = StickNav()
        self.menu = flag

    @staticmethod
    def available() -> bool:
        try:
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            return pygame.joystick.get_count() > 0
        except pygame.error:
            return False

    @property
    def name(self) -> str:
        try:
            return self.joy.get_name()
        except Exception:               # a stub need not implement it
            return "gamepad"

    def _axis(self, i: int) -> float:
        if i is None or i < 0:
            return 0.0
        try:
            return float(self.joy.get_axis(i))
        except Exception:
            return 0.0

    def _button(self, i: int) -> bool:
        try:
            if self.layout == "ps" and 11 <= i <= 14 and i >= self.joy.get_numbuttons():
                # some drivers report the d-pad as hat 0 instead of buttons
                hx, hy = self.joy.get_hat(0)
                return {11: hy == 1, 12: hy == -1, 13: hx == -1, 14: hx == 1}[i]
            return bool(self.joy.get_button(i))
        except Exception:
            return False

    def _trigger(self, key: str) -> float:
        """One trigger axis as 0..1, normalised from its LATCHED rest value.

        `_trig_rest[key]` is -1.0 for a well-behaved driver, which makes this
        the historical (a + 1) / 2 exactly. It is set to 0.0 only when the
        pad's FIRST report showed that axis sitting at 0 rather than -1, which
        is the other convention some macOS drivers use; anything else in
        between keeps -1.0 and `_check_rest` has already printed the warning.
        """
        rest = self._trig_rest.get(key, -1.0)
        a = self._axis(self.map[key])
        return _clamp(_deadzone((a - rest) / (1.0 - rest), PAD_DEADZONE_TRIG),
                      0.0, 1.0)

    def _check_rest(self) -> None:
        """One-line warning if a trigger axis does not rest at -1: the sign
        that the OS gave SDL a different axis order than the layout assumes.

        Deferred until the pad has actually reported: right after attach every
        axis reads exactly 0.0 (no HID report yet), which looked like "trigger
        rests at 0" and warned on a correctly mapped DualSense. Measured on
        the hardware: the first report arrives within a frame and both
        triggers read -1.0."""
        try:
            n = self.joy.get_numaxes()
        except Exception:
            n = 0
        if n and all(self._axis(k) == 0.0 for k in range(n)):
            self._rest_checked = False      # nothing reported yet; try again
            return
        self._rest_checked = True
        warn = None
        for key in ("throttle", "brake"):
            i = self.map.get(key, -1)
            if i is None or i < 0:
                continue
            v = self._axis(i)
            # LATCH FIRST, warn afterwards: the old spelling returned on the
            # first oddity, so the second trigger was never looked at.
            if abs(v + 1.0) <= 0.25:
                self._trig_rest[key] = -1.0      # SDL / DualSense: rest -1
            elif abs(v) <= 0.25:
                self._trig_rest[key] = 0.0       # the other convention: rest 0
                print(f"gamepad: {key} axis {i} rests at {v:+.2f}, not -1.0 - "
                      f"reading it as a 0..+1 trigger")
            elif warn is None:
                warn = (key, i, v)               # mid-travel: really wrong
        if warn is not None:
            key, i, v = warn
            print(f"gamepad: {key} axis {i} rests at {v:+.2f}, expected -1.0 "
                  f"- the {self.layout} layout may not match this driver; "
                  f"run `python3 -m drive.drive --pad-calib` and write "
                  f"~/.carsim_pad.json")

    # -- named access (the garage editor and the calib printout use these) --
    def pressed(self, name: str) -> bool:
        i = self._name_to_btn.get(name)
        return False if i is None else self._button(i)

    def stick(self, which: str = "left") -> tuple[float, float]:
        """Post-deadzone (x, y) of a stick, SDL sign: +x right, +y DOWN."""
        if which == "left":
            ix, iy = self.map.get("steer", 0), self.map.get("ly", 1)
        else:
            ix, iy = self.map.get("rx", 2), self.map.get("ry", 3)
        return (_deadzone(self._axis(ix), PAD_DEADZONE_STICK),
                _deadzone(self._axis(iy), PAD_DEADZONE_STICK))

    def button_name(self, i: int) -> str:
        return self.names.get(i, f"b{i}")

    # -- haptics --------------------------------------------------------------
    def rumble(self, low: float, high: float, ms: int = 120) -> None:
        """Best effort; a pad that cannot rumble is silently left alone."""
        if not PAD_RUMBLE or self._rumble_ok is False:
            return
        low, high = _clamp(low, 0.0, 1.0), _clamp(high, 0.0, 1.0)
        try:
            if low <= 0.0 and high <= 0.0:
                if self._rumble_last != (0.0, 0.0):
                    self.joy.stop_rumble()
            else:
                ok = self.joy.rumble(low, high, int(ms))
                if self._rumble_ok is None:
                    self._rumble_ok = bool(ok)
            self._rumble_last = (low, high)
        except Exception:
            self._rumble_ok = False

    def feedback(self, aux) -> None:
        """Haptics from the frame's HudData, at PAD_RUMBLE_HZ.

        Low motor: tyre utilisation above 0.85 (the limit is coming), and a
        constant rumble on the grass. High motor: a wheel locked, spinning or
        in the air. Called from the RENDER loop only - it reads a wall clock,
        which the physics path must never do.
        """
        now = time.monotonic()
        if now - self._rumble_t < 1.0 / PAD_RUMBLE_HZ:
            return
        self._rumble_t = now
        low = high = 0.0
        try:
            util = max(float(aux.util_f), float(aux.util_r))
            if util > 0.85:
                low = 0.65 * min(1.0, (util - 0.85) / 0.15)
            slipping = (any(abs(float(k)) > 0.12 for k in aux.kappa)
                        or any(bool(w) for w in aux.wheel_lift))
            if slipping and float(aux.V) > 1.0:
                high = 0.7
            if not aux.on_track and float(aux.V) > 2.0:
                low = max(low, 0.35)
            if getattr(aux, "paused", False):
                low = high = 0.0
        except (AttributeError, TypeError, ValueError):
            low = high = 0.0
        self.rumble(low, high, int(1500.0 / PAD_RUMBLE_HZ))

    def poll_events(self) -> list[str]:
        """Button EDGES. Read from the device state, not the queue, so that a
        BlendedInput's keyboard half can own the queue drain without either
        half stealing the other's events."""
        cmds: list[str] = []
        if self.menu:
            return self._poll_menu()
        for btn, cmd in self.map["buttons"].items():
            now = self._button(btn)
            was = self._buttons_prev.get(btn, False)
            self._buttons_prev[btn] = now
            if now and not was and cmd not in ("handbrake", "clutch"):
                cmds.append(cmd)
                if cmd == "shift_up":
                    self._pending_gear = +1
                elif cmd == "shift_down":
                    self._pending_gear = -1
                # 'wing' / 'wing_side' are NOT folded into pad state: drive.py's
                # Sim owns that toggle and ORs it into Controls. A second copy
                # here made circle-then-F leave the panel armed with both
                # readouts saying OFF.
        return cmds

    def _poll_menu(self) -> list[str]:
        """Menu mode: named nav edges + left-stick up/down. The driving edge
        state is refreshed silently so a button still held when the menu
        closes (circle = back, then circle = wing) cannot fire on the way out."""
        cmds: list[str] = []
        table = MENU_PAD_NAMES.get(self.layout, MENU_PAD_NAMES["generic"])
        for name, cmd in table.items():
            now = self.pressed(name)
            was = self._menu_prev.get(name, False)
            self._menu_prev[name] = now
            if now and not was:
                cmds.append(cmd)
        if self._menu_stick is not None:
            nav = self._menu_stick.poll(self.stick("left")[1])
            if nav:
                cmds.append(nav)
        for btn in self.map["buttons"]:
            self._buttons_prev[btn] = self._button(btn)
        return cmds

    def update(self, dt: float, V: float = 0.0, beta_deg: float = 0.0,
               rpm: float = 0.0, gear: int = 1) -> Controls:
        if not self._rest_checked:
            self._check_rest()             # first report after attach
        s = _deadzone(self._axis(self.map["steer"]), PAD_DEADZONE_STICK)
        self.axis_steer = s
        g = math.copysign(abs(s) ** PAD_EXPO, s)        # expo: fine near centre

        # The same directional soft lock the keyboard uses (see
        # steer_limit_pair_deg and KeyboardInput.update): the beta bonus is
        # spent on the counter-steer side only. On a pad this matters twice
        # over, because the stick is PROPORTIONAL to the bound -- with the old
        # symmetric bonus the stick's gain grew as the car slid, so the same
        # stick position meant a different road-wheel angle from one moment to
        # the next. Now full stick into the slide means the floor and full
        # stick out of it means the catch.
        if self.steer_limit:
            lim_l, lim_r = steer_limit_pair_deg(V, beta_deg,
                                                k_us_deg=self.k_us_deg)
        else:
            lim_l = lim_r = DELTA_LOCK_DEG
        ease = _return_rate_deg(V) * dt
        self._lim_l = lim_l if lim_l >= self._lim_l else max(lim_l, self._lim_l - ease)
        self._lim_r = lim_r if lim_r >= self._lim_r else max(lim_r, self._lim_r - ease)
        self.delta_lim_deg = self._lim_r if g > 0.0 else self._lim_l
        # SDL axis0 is -1 at full LEFT stick, +1 at full RIGHT, while the
        # contract's delta is +ve LEFT. The sign flip lives here, once.
        target = -(self._lim_r if g > 0.0 else self._lim_l) * g
        dmax = PAD_RATE_DEG * dt
        self.delta_deg += _clamp(target - self.delta_deg, -dmax, dmax)
        self.delta_deg = _clamp(self.delta_deg, -self._lim_r, self._lim_l)

        # Triggers rest at -1 and travel to +1, hence the (a+1)/2 that
        # `_trigger` generalises. TWO traps live here and both are now closed:
        #   * before the pad's first HID report EVERY axis reads exactly 0.0,
        #     and (0+1)/2 is HALF TRAVEL -- the old code handed the car 47%
        #     throttle and 47% brake simultaneously on hot-plug, for however
        #     many frames it took the pad to report. `_rest_checked` already
        #     detects that state for its warning; now it also gates the pedals.
        #   * a driver that rests its triggers at 0.0 and travels to +1.0 (the
        #     other convention in the wild) read as 50-100% instead of 0-100%.
        #     `_trig_rest` latches whichever rest the hardware actually showed
        #     on its first report and normalises from there, so both
        #     conventions give 0 at rest and 1 at the stop.
        if not self._rest_checked:
            self.throttle = self.brake = 0.0
        else:
            if self.map["throttle"] >= 0:
                self.throttle = self._trigger("throttle")
            else:
                self.throttle = 1.0 if self._button(7) else 0.0  # no trigger axes
            if self.map["brake"] >= 0:
                self.brake = self._trigger("brake")
            else:
                self.brake = 1.0 if self._button(6) else 0.0

        btns = self.map["buttons"]
        self.clutch = 1.0 if any(self._button(b) for b, c in btns.items() if c == "clutch") else 0.0
        self.handbrake = 1.0 if any(self._button(b) for b, c in btns.items() if c == "handbrake") else 0.0

        gear_req, self._pending_gear = self._pending_gear, 0
        return Controls(delta=math.radians(self.delta_deg), throttle=self.throttle,
                        brake=self.brake, clutch=self.clutch, handbrake=self.handbrake,
                        gear_req=gear_req, auto_gearbox=self.auto_gearbox,
                        auto_clutch=self.auto_clutch,
                        wing_on=self.wing_on, starter=False)


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------


class BlendedInput:
    """Keyboard + optional pad. Per axis, the pad wins above |0.02|.

    With `pad=None` this is exactly KeyboardInput plus one printed line, which
    is the DEFAULT path on this machine (V29): joystick count is 0, so any
    code that assumed a pad object exists would dereference None on the first
    frame. Nothing here may assume `self.pad` is not None.
    """

    HOTPLUG_EVERY = 30              # render frames between joystick recounts

    def __init__(self, kb: KeyboardInput, pad: Optional[GamepadInput] = None,
                 announce: bool = True, hotplug: bool = True):
        self.kb = kb
        self.pad = pad
        self.hotplug = bool(hotplug)
        self._frames = 0
        if announce:
            if pad is None:
                print("no gamepad found - keyboard only (a pad can still be "
                      "connected while driving)")
            else:
                print(f"gamepad: {pad.name} ({pad.layout} layout)")

    def _hotplug(self) -> None:
        """Attach a pad that appears after launch; drop one that goes away.
        pygame only refreshes the count while events are pumped, which the
        keyboard half's queue drain does every frame."""
        try:
            n = pygame.joystick.get_count() if pygame.joystick.get_init() else 0
            if n == 0 and not pygame.joystick.get_init():
                pygame.joystick.init()
                n = pygame.joystick.get_count()
        except pygame.error:
            return
        if self.pad is None and n > 0:
            try:
                self.pad = GamepadInput(0, steer_limit=self.kb.steer_limit,
                                        k_us_deg=self.kb.k_us_deg)
                print(f"gamepad connected: {self.pad.name} ({self.pad.layout} layout)")
                if self.kb.menu:
                    self.pad.set_menu(True)
            except Exception as exc:
                print(f"gamepad appeared but is not usable ({exc})")
                self.hotplug = False
        elif self.pad is not None and n == 0:
            print("gamepad disconnected - keyboard only")
            self.pad = None

    def feedback(self, aux) -> None:
        if self.pad is not None:
            self.pad.feedback(aux)

    def set_menu(self, flag: bool) -> None:
        self.kb.set_menu(flag)
        if self.pad is not None:
            self.pad.set_menu(flag)

    def set_gearbox(self, mode: str) -> None:
        self.kb.set_gearbox(mode)
        if self.pad is not None:
            self.pad.set_gearbox(mode)

    def set_steer_limit(self, flag: bool) -> None:
        self.kb.steer_limit = bool(flag)
        if self.pad is not None:
            self.pad.steer_limit = bool(flag)

    @property
    def gearbox(self) -> str:
        for m in GEARBOX_MODES:
            if gearbox_flags(m) == (self.kb.auto_gearbox, self.kb.auto_clutch):
                return m
        return "auto"

    @property
    def menu(self) -> bool:
        return self.kb.menu

    @property
    def layout(self) -> str | None:
        """The attached pad's layout ('ps' | 'generic'), None without one."""
        return self.pad.layout if self.pad is not None else None

    # Keep the HUD-facing state readable through the blend.
    @property
    def wing_on(self) -> bool:
        return self.kb.wing_on or (self.pad.wing_on if self.pad else False)

    @property
    def wing_side_mode(self) -> str:
        return self.kb.wing_side_mode

    @property
    def delta_lim_deg(self) -> float:
        return self.kb.delta_lim_deg

    @property
    def steer_limit(self) -> bool:
        return self.kb.steer_limit

    @property
    def paused(self) -> bool:
        return self.kb.paused

    def poll_events(self) -> list[str]:
        cmds = self.kb.poll_events()
        self._frames += 1
        if self.hotplug and self._frames % self.HOTPLUG_EVERY == 1:
            self._hotplug()
        if self.pad is not None:
            cmds.extend(self.pad.poll_events())
        return cmds

    def update(self, dt: float, V: float = 0.0, beta_deg: float = 0.0,
               rpm: float = 0.0, gear: int = 1) -> Controls:
        c = self.kb.update(dt, V, beta_deg, rpm, gear)
        if self.pad is None:
            return c
        p = self.pad.update(dt, V, beta_deg, rpm, gear)
        pick = lambda a, b: a if abs(a) > PAD_BLEND_EPS else b   # noqa: E731
        # Steer is the one axis where the spec's "|pad_value| > 0.02" cannot be
        # read literally: p.delta is an ANGLE IN RADIANS, so a 0.02 threshold on
        # it would silently ignore the first 1.15 deg of every stick input. The
        # test is made on the stick's own normalised travel instead, plus a
        # hold while the rate-capped angle is still slewing back to centre.
        pad_steers = (abs(self.pad.axis_steer) > PAD_BLEND_EPS
                      or abs(self.pad.delta_deg) > 0.05)
        return Controls(
            delta=p.delta if pad_steers else c.delta,
            throttle=pick(p.throttle, c.throttle),
            brake=pick(p.brake, c.brake),
            clutch=pick(p.clutch, c.clutch),
            handbrake=pick(p.handbrake, c.handbrake),
            gear_req=p.gear_req if p.gear_req else c.gear_req,
            auto_gearbox=c.auto_gearbox,
            auto_clutch=c.auto_clutch,
            wing_on=c.wing_on or p.wing_on,
            starter=c.starter,
        )

    def reset(self) -> None:
        self.kb.reset()


# ---------------------------------------------------------------------------
# Scripted
# ---------------------------------------------------------------------------


class ScriptedInput:
    """A closure in the driver's seat. No pygame, no wall clock, ever.

    `fn(t, veh, track) -> Controls` is called once per physics step with
    `t = n_steps*dt` (contract section 0: never an accumulated float, so a
    replay is bit-identical). The returned Controls is passed through
    UNTOUCHED - no ramp, no limiter, no deadzone - which is the whole point:
    every acceptance script commands delta directly and the driver aid must
    never be able to clip a measured number.
    """

    def __init__(self, fn: Callable[[float, object, object], Controls],
                 vehicle=None, track=None):
        self.fn = fn
        self.vehicle = vehicle
        self.track = track
        self.n_steps = 0
        self.dt = 0.0

    def bind(self, vehicle, track=None) -> "ScriptedInput":
        self.vehicle = vehicle
        self.track = track
        return self

    @property
    def t(self) -> float:
        return self.n_steps * self.dt

    def poll_events(self) -> list[str]:
        return []

    def update(self, dt: float, V: float = 0.0, beta_deg: float = 0.0,
               rpm: float = 0.0, gear: int = 1) -> Controls:
        self.dt = dt
        t = self.n_steps * dt
        ctl = self.fn(t, self.vehicle, self.track)
        self.n_steps += 1
        return ctl

    def reset(self) -> None:
        self.n_steps = 0


def default_input(steer_limit: bool = True, announce: bool = True,
                  hotplug: bool = True, **kb_kwargs) -> BlendedInput:
    """What drive.py should construct: pad if one exists, keyboard if not.

    Not in the contract's API list; it exists so the None-pad case is written
    once instead of in drive.py and in validate.py separately (V29).
    """
    kb = KeyboardInput(steer_limit=steer_limit, **kb_kwargs)
    pad = None
    if GamepadInput.available():
        try:
            pad = GamepadInput(0, steer_limit=steer_limit, k_us_deg=kb.k_us_deg)
        except Exception as exc:        # a pad that enumerates but will not open
            print(f"gamepad found but not usable ({exc}) - keyboard only")
            pad = None
    return BlendedInput(kb, pad, announce=announce, hotplug=hotplug)


# ---------------------------------------------------------------------------
# Self-check: specs/harness.txt V15-V18, V29, plus the delta_lim(V) table.
# ---------------------------------------------------------------------------

DT = 0.001                              # DT_PHYS; every ramp integrates here


def _hold(inp: KeyboardInput, T: float, V: float = 0.0, beta_deg: float = 0.0,
          dt: float = DT, **keys):
    """Hold a key state for T seconds of PHYSICS and return the trace.

    Every key not named is explicitly RELEASED - a partial update would leave
    the previous call's keys latched, which silently turns a release test into
    a both-arrows test (d_in = 0 either way, and the bug hides).
    """
    state = {k: False for k in inp.held}
    state.update(keys)
    inp.set_keys(**state)
    n = int(round(T / dt))
    trace = []
    for _ in range(n):
        c = inp.update(dt, V, beta_deg, 0.0, 3)
        trace.append(c)
    return trace


def _first_time(trace, pred, dt: float = DT) -> float:
    for i, c in enumerate(trace):
        if pred(c):
            return (i + 1) * dt
    return float("nan")


def self_check(verbose: bool = True) -> bool:
    ok = True
    fails: list[str] = []

    def check(name: str, got, exp, tol, unit=""):
        nonlocal ok
        good = abs(got - exp) <= tol
        if not good:
            ok = False
            fails.append(name)
        if verbose:
            print(f"  {'PASS' if good else 'FAIL'}  {name:<52s} "
                  f"{got:11.5f} vs {exp:10.5f} +/-{tol:g} {unit}")

    def check_eq(name: str, got, exp):
        nonlocal ok
        good = got == exp
        if not good:
            ok = False
            fails.append(name)
        if verbose:
            print(f"  {'PASS' if good else 'FAIL'}  {name:<52s} {got!r} vs {exp!r}")

    if verbose:
        print("drive/input.py self-check  (specs/harness.txt V15-V18, V29)")
        print(f"  road-wheel drive rate {W_DRIVE_DEG:.2f} deg/s, lock "
              f"{DELTA_LOCK_DEG} deg -> {DELTA_LOCK_DEG / W_DRIVE_DEG:.3f} s to lock")

        print("\n-- steer_limit_deg(V, 0): the delta_lim(V) table --")
    table = [(0.0, 32.625), (6.87, 32.597), (8.0, 24.87), (10.0, 17.06),
             (15.0, 9.35), (20.0, 6.65), (20.568, 6.457), (25.0, 5.396),
             (29.087, 4.815), (30.0, 4.717), (40.0, 4.042), (47.2, 3.797)]
    for V, exp in table:
        check(f"delta_lim({V:g} m/s)", steer_limit_deg(V, 0.0), exp, 0.01, "deg")

    if verbose:
        print("\n-- V17 the limiter opens up in a slide --")
    check("V17 delta_lim_eff(30 m/s, beta=12 deg)", steer_limit_deg(30.0, 12.0),
          19.117, 0.05, "deg")
    check("V17 delta_lim_eff(30 m/s, beta=0)", steer_limit_deg(30.0, 0.0),
          4.717, 0.05, "deg")
    check("V17 beta cannot exceed mechanical lock", steer_limit_deg(30.0, 90.0),
          32.625, 1e-9, "deg")

    if verbose:
        print("\n-- V15 steering ramp at rest, holding RIGHT --")
    kb = KeyboardInput(steer_limit=True)
    tr = _hold(kb, 1.0, V=0.0, right=True)
    d200 = math.degrees(tr[199].delta)
    check("V15 |delta| at t = 0.200 s", abs(d200), 11.25, 0.01, "deg")
    check_eq("V15 RIGHT gives delta < 0 (contract: +delta = LEFT)", d200 < 0.0, True)
    t_lock = _first_time(tr, lambda c: abs(math.degrees(c.delta)) >= DELTA_LOCK_DEG - 1e-9)
    check("V15 time to full lock", t_lock, 0.580, 0.005, "s")
    check("V15 clamped at lock thereafter", abs(math.degrees(tr[-1].delta)),
          32.625, 1e-9, "deg")

    if verbose:
        print("\n-- V16 limited at 25 m/s, then released --")
    kb = KeyboardInput(steer_limit=True)
    tr = _hold(kb, 2.0, V=25.0, right=True)
    d_sat = abs(math.degrees(tr[-1].delta))
    check("V16 saturated |delta| at V = 25 m/s", d_sat, 5.396, 0.05, "deg")
    tr = _hold(kb, 0.5, V=25.0)          # release: no direction key held
    t_ret = _first_time(tr, lambda c: c.delta == 0.0)
    check("V16 return to centre (5.396 / 90 deg/s)", t_ret, 0.060, 0.005, "s")
    check("V16 return rate at 25 m/s", _return_rate_deg(25.0), 90.0, 1e-9, "deg/s")
    check("V16 return rate at rest", _return_rate_deg(0.0), 45.0, 1e-9, "deg/s")

    if verbose:
        print("\n-- V18 pedal ramps --")
    kb = KeyboardInput()
    tr = _hold(kb, 0.5, up=True)
    check("V18 throttle 0 -> 1", _first_time(tr, lambda c: c.throttle >= 1.0),
          0.2857, 0.001, "s")
    tr = _hold(kb, 0.5, up=False)
    check("V18 throttle 1 -> 0", _first_time(tr, lambda c: c.throttle <= 0.0),
          0.1667, 0.001, "s")
    kb = KeyboardInput()
    tr = _hold(kb, 0.5, down=True)
    check("V18 brake 0 -> 1", _first_time(tr, lambda c: c.brake >= 1.0),
          0.2000, 0.001, "s")
    tr = _hold(kb, 0.5, down=False)
    check("V18 brake 1 -> 0", _first_time(tr, lambda c: c.brake <= 0.0),
          0.1250, 0.001, "s")
    kb = KeyboardInput()
    tr = _hold(kb, 1.0, up=True, down=True, fine=True)
    check("V18 LSHIFT caps throttle", tr[-1].throttle, 0.500, 1e-12)
    check("V18 LSHIFT caps brake", tr[-1].brake, 0.500, 1e-12)
    tr = _hold(kb, 0.5, up=True, down=True, fine=False)
    check("V18 releasing LSHIFT ramps on (no jump)", tr[0].throttle,
          0.5 + RATE_THROTTLE[0] * DT, 1e-12)
    kb = KeyboardInput()
    tr = _hold(kb, 0.4, clutch=True)
    check("clutch 0 -> 1 (8.0 /s)", _first_time(tr, lambda c: c.clutch >= 1.0),
          0.1250, 0.001, "s")
    tr = _hold(kb, 0.6, clutch=False)
    check("clutch 1 -> 0 (3.0 /s)", _first_time(tr, lambda c: c.clutch <= 0.0),
          0.3333, 0.001, "s")
    kb = KeyboardInput()
    tr = _hold(kb, 0.4, handbrake=True)
    check("handbrake 0 -> 1 (8.0 /s)", _first_time(tr, lambda c: c.handbrake >= 1.0),
          0.1250, 0.001, "s")

    if verbose:
        print("\n-- steering: fine, counter-steer, and the bypass --")
    kb = KeyboardInput()
    tr = _hold(kb, 0.2, right=True, fine=True)
    check("FINE halves the steer rate", abs(math.degrees(tr[-1].delta)),
          11.25 / 2, 0.01, "deg")
    kb = KeyboardInput()
    _hold(kb, 0.2, V=10.0, right=True)                  # to -11.25 deg
    d0 = kb.delta_deg
    tr = _hold(kb, 0.05, V=10.0, left=True)             # flick back
    rate = (kb.delta_deg - d0) / 0.05
    check("counter-steer rate at 10 m/s (drive+return)", rate,
          W_DRIVE_DEG + _return_rate_deg(10.0), 0.2, "deg/s")
    kb = KeyboardInput(steer_limit=False)
    tr = _hold(kb, 1.0, V=30.0, right=True)
    check("steer_limit=False bypasses the aid at 30 m/s",
          abs(math.degrees(tr[-1].delta)), 32.625, 1e-9, "deg")
    check_eq("steer_limit=False reports lock as the limit",
             round(kb.delta_lim_deg, 6), 32.625)
    # V17 through the CLAMP, and the sign pairing matters now that the bonus is
    # directional. beta = atan2(v, |u|) with v leftward, so beta < 0 is a car
    # whose velocity has swung to the RIGHT of its nose -- a left-turn slide --
    # and the catch is RIGHT lock. That is the pairing the aid must serve, and
    # it still gets the full V17 19.12 deg. The old spelling of this check held
    # beta = +12 with RIGHT lock, which is lock INTO the slide; it passed only
    # because the bonus used to be symmetric, and that symmetry is the bug
    # (.handoff/08-steering.md, symptom c).
    kb = KeyboardInput(steer_limit=True)
    tr = _hold(kb, 1.0, V=30.0, beta_deg=-12.0, right=True)
    check("V17 a 12 deg slide gives 19.12 deg of COUNTER lock",
          abs(math.degrees(tr[-1].delta)), 19.117, 0.05, "deg")
    kb = KeyboardInput(steer_limit=True)
    tr = _hold(kb, 1.0, V=30.0, beta_deg=-12.0, left=True)
    check("V17 the same slide does NOT open lock into the slide",
          abs(math.degrees(tr[-1].delta)), 4.717, 0.05, "deg")
    check_eq("steer_limit_pair_deg is symmetric at beta = 0",
             steer_limit_pair_deg(30.0, 0.0),
             (steer_limit_deg(30.0, 0.0), steer_limit_deg(30.0, 0.0)))
    check("steer_limit_pair_deg counter side == steer_limit_deg",
          steer_limit_pair_deg(30.0, -12.0)[1], steer_limit_deg(30.0, 12.0),
          1e-12, "deg")
    # the bound eases DOWN rather than snapping: wind on 19.12 deg of catch,
    # then let beta collapse to zero and watch the wheel unwind at the return
    # rate instead of teleporting to the 4.717 deg floor.
    d_caught = kb.delta_deg
    kb2 = KeyboardInput(steer_limit=True)
    _hold(kb2, 1.0, V=30.0, beta_deg=-12.0, right=True)
    tr = _hold(kb2, 0.001, V=30.0, beta_deg=0.0, right=True)
    check("the soft lock eases down, it does not snap (one step)",
          abs(math.degrees(tr[-1].delta)), 19.117 - _return_rate_deg(30.0) * DT,
          0.02, "deg")
    tr = _hold(kb2, 0.20, V=30.0, beta_deg=0.0, right=True)
    check("... and reaches the floor in 19.12-4.72 / 90 s",
          abs(math.degrees(tr[-1].delta)), 4.717, 0.05, "deg")

    if verbose:
        print("\n-- edge triggering (E/Q/F/P from the event queue only) --")
    # SDL reads SDL_VIDEODRIVER at display.init(), not at import, so the
    # self-check can force the dummy driver here and never open a window even
    # when CARSIM_HEADLESS was not exported. setdefault: an explicit choice wins.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.display.init()
    pygame.display.set_mode((320, 200))
    kb = KeyboardInput()
    check_eq("headless SDL driver", pygame.display.get_driver(), "dummy")
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e, mod=0))
    cmds = kb.poll_events()
    check_eq("E -> 'shift_up'", "shift_up" in cmds, True)
    gears = [kb.update(DT, 20.0, 0.0, 3000.0, 3).gear_req for _ in range(1000)]
    check_eq("one press = exactly one gear_req in 1000 steps",
             (sum(1 for g in gears if g != 0), gears[0]), (1, +1))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q, mod=0))
    kb.poll_events()
    check_eq("Q -> gear_req -1", kb.update(DT, 20.0, 0.0, 3000.0, 3).gear_req, -1)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
    kb.poll_events()
    check_eq("F toggles wing_on", kb.update(DT).wing_on, True)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
    kb.poll_events()
    check_eq("F again toggles it back", kb.update(DT).wing_on, False)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0))
    kb.poll_events()
    check_eq("G cycles the wing side", kb.wing_side_mode, "left")
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r,
                                         mod=pygame.KMOD_LSHIFT))
    check_eq("SHIFT+R -> 'full_reset'", kb.poll_events(), ["full_reset"])
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=0))
    check_eq("R -> 'reset'", kb.poll_events(), ["reset"])
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    check_eq("window close -> 'quit'", kb.poll_events(), ["quit"])

    if verbose:
        print("\n-- pause menu: ESC opens, nav keys only while open --")
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    check_eq("ESC -> 'menu' (not quit)", kb.poll_events(), ["menu"])
    kb.set_menu(True)
    kb.set_keys(up=True)                   # throttle held when the menu opens
    for key, want in ((pygame.K_UP, "nav_up"), (pygame.K_DOWN, "nav_down"),
                      (pygame.K_RETURN, "select"), (pygame.K_SPACE, "select"),
                      (pygame.K_r, "reset"), (pygame.K_BACKSPACE, "garage"),
                      (pygame.K_p, "menu"), (pygame.K_ESCAPE, "menu")):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        check_eq(f"menu: {pygame.key.name(key)} -> {want!r}", kb.poll_events(), [want])
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r,
                                         mod=pygame.KMOD_LSHIFT))
    check_eq("menu: SHIFT+R -> 'full_reset'", kb.poll_events(), ["full_reset"])
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e, mod=0))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
    check_eq("menu: E / F swallowed (no shift, no wing)",
             (kb.poll_events(), kb.update(DT).gear_req, kb.wing_on), ([], 0, False))
    check_eq("menu: held keys read released", any(kb.held.values()), False)
    kb.set_menu(False)
    kb.set_keys(clear=True)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    check_eq("menu closed: UP is a pedal again, not nav", kb.poll_events(), [])

    if verbose:
        print("\n-- V29 gamepad absent: the keyboard-only path --")
    if GamepadInput.available():
        # a real pad is plugged in: the absent-pad assertions are not testable,
        # the None-safety of the blend is exercised with pad=None instead.
        print(f"  (a gamepad is connected: {pygame.joystick.Joystick(0).get_name()}; "
              f"V29 'absent' checks run on a forced-None pad)")
        inp = BlendedInput(KeyboardInput(steer_limit=True), None, announce=True,
                           hotplug=False)
    else:
        check_eq("GamepadInput.available()", GamepadInput.available(), False)
        check_eq("pygame.joystick.get_count() == 0", pygame.joystick.get_count(), 0)
        inp = default_input(steer_limit=True, announce=True)
        inp.hotplug = False
    check_eq("default_input() gives a live keyboard, pad None", inp.pad, None)
    inp.kb.set_keys(up=True, right=True)
    c = None
    for _ in range(500):
        c = inp.update(DT, 15.0, 0.0, 3000.0, 2)        # no None dereference
    check("V29 driveable: throttle after 0.5 s", c.throttle, 1.0, 1e-12)
    check("V29 driveable: |delta| after 0.5 s", abs(math.degrees(c.delta)),
          min(9.3458, 0.5 * W_DRIVE_DEG), 0.02, "deg")
    check_eq("V29 poll_events with no pad", inp.poll_events(), [])

    if verbose:
        print("\n-- gamepad axis maths (stub device; no hardware here) --")

    class _StubPad:
        """Six axes, SDL2 layout. Sticks centred, triggers at rest (-1)."""
        def __init__(self):
            self.ax = [0.0, 0.0, -1.0, 0.0, -1.0, -1.0]
            self.bt = [False] * 8

        def get_numaxes(self):
            return len(self.ax)

        def get_axis(self, i):
            return self.ax[i]

        def get_numbuttons(self):
            return len(self.bt)

        def get_button(self, i):
            return self.bt[i]

        def get_name(self):
            return "stub"

    stub = _StubPad()
    pad = GamepadInput(joystick=stub, steer_limit=False, user_config=False)
    check_eq("stub -> generic layout", pad.layout, "generic")
    c = pad.update(DT, 10.0)
    check("pad centred -> zero steer", c.delta, 0.0, 1e-12)
    check("pad triggers at rest -> zero throttle", c.throttle, 0.0, 1e-12)
    stub.ax[0] = 0.10
    c = pad.update(DT, 10.0)
    check("pad inside the 0.12 deadzone -> zero", c.delta, 0.0, 1e-12)
    stub.ax[0] = 1.0
    stub.ax[5] = 1.0
    for _ in range(1000):
        c = pad.update(DT, 10.0)
    check("pad rate cap 250 deg/s reaches lock in 0.131 s",
          abs(math.degrees(c.delta)), 32.625, 1e-6, "deg")
    check("pad trigger fully pressed -> throttle 1", c.throttle, 1.0, 1e-12)
    stub.ax[0] = 0.56                    # dz -> 0.5, expo 1.5 -> 0.35355
    pad.delta_deg = 0.0
    c = pad.update(1.0, 10.0)            # a big dt so the rate cap does not bind
    check("pad expo 1.5 at half stick", abs(math.degrees(c.delta)),
          32.625 * 0.5 ** 1.5, 1e-6, "deg")
    check_eq("stick RIGHT gives delta < 0 (contract: +delta = LEFT)",
             c.delta < 0.0, True)
    stub.bt[0] = True
    check("pad button 0 = handbrake", pad.update(DT, 10.0).handbrake, 1.0, 1e-12)
    stub.bt[5] = True
    pad.poll_events()
    check_eq("pad button 5 = shift up (edge)", pad.update(DT, 10.0).gear_req, +1)
    check_eq("pad button held is not a second shift",
             pad.update(DT, 10.0).gear_req, 0)

    if verbose:
        print("\n-- blend: the pad wins an axis above |0.02| --")
    kb = KeyboardInput()
    stub = _StubPad()
    pad = GamepadInput(joystick=stub, steer_limit=False, user_config=False)
    bl = BlendedInput(kb, pad, announce=False, hotplug=False)
    kb.set_keys(up=True)
    for _ in range(200):
        c = bl.update(DT, 10.0)
    check("keyboard owns the pedal while the trigger rests", c.throttle,
          min(1.0, 0.2 * RATE_THROTTLE[0]), 1e-12)
    stub.ax[5] = 0.0                     # trigger at half travel -> 0.5
    c = bl.update(DT, 10.0)
    check("pad takes the pedal once it moves", c.throttle,
          _deadzone(0.5, PAD_DEADZONE_TRIG), 1e-12)
    kb.set_keys(left=True)
    stub.ax[0] = 0.20                    # dz -> 0.0909, expo -> 0.0274 of lock
    for _ in range(50):
        c = bl.update(DT, 10.0)
    check_eq("a small stick input still beats the keyboard on steer",
             c.delta < 0.0 and abs(c.delta) < 0.02, True)

    if verbose:
        print("\n-- PlayStation layout (DualSense stub: 6 axes, 16 buttons) --")

    class _StubPS(_StubPad):
        def __init__(self):
            self.ax = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0]
            self.bt = [False] * 16
            self.hat = (0, 0)

        def get_name(self):
            return "DualSense Wireless Controller"

        def get_numhats(self):
            return 0

    ps = _StubPS()
    pad = GamepadInput(joystick=ps, steer_limit=False, user_config=False)
    check_eq("DualSense name -> ps layout", pad.layout, "ps")
    check_eq("ps: R2 is throttle (axis 5)", pad.map["throttle"], 5)
    check_eq("ps: L2 is brake (axis 4)", pad.map["brake"], 4)
    ps.ax[5] = 1.0
    ps.ax[4] = 0.0
    c = pad.update(DT, 10.0)
    check("ps: R2 full -> throttle 1", c.throttle, 1.0, 1e-12)
    check("ps: L2 half -> brake 0.5", c.brake, _deadzone(0.5, PAD_DEADZONE_TRIG), 1e-12)
    ps.bt[10] = True
    pad.poll_events()
    check_eq("ps: R1 = shift up", pad.update(DT, 10.0).gear_req, +1)
    ps.bt[10] = False
    ps.bt[9] = True
    pad.poll_events()
    check_eq("ps: L1 = shift down", pad.update(DT, 10.0).gear_req, -1)
    ps.bt[9] = False
    ps.bt[0] = True
    check("ps: cross = handbrake", pad.update(DT, 10.0).handbrake, 1.0, 1e-12)
    ps.bt[2] = True
    check("ps: square = clutch", pad.update(DT, 10.0).clutch, 1.0, 1e-12)
    ps.bt[0] = ps.bt[2] = False
    ps.bt[15] = True
    check_eq("ps: touchpad -> 'garage'", "garage" in pad.poll_events(), True)
    ps.bt[15] = False
    ps.bt[1] = True
    ev = pad.poll_events()
    check_eq("ps: circle -> 'wing' edge, state NOT folded into the pad",
             ("wing" in ev, pad.wing_on), (True, False))
    ps.bt[1] = False
    ps.ax[2], ps.ax[3] = 0.5, -0.5
    check_eq("ps: right stick readable by name",
             tuple(round(v, 4) for v in pad.stick("right")),
             (round(_deadzone(0.5, PAD_DEADZONE_STICK), 4),
              round(_deadzone(-0.5, PAD_DEADZONE_STICK), 4)))
    ps.bt[3] = True
    check_eq("ps: pressed('triangle')", pad.pressed("triangle"), True)
    pad.rumble(0.5, 0.5, 50)               # stub has no rumble: must not raise
    check_eq("rumble on a pad without haptics is a no-op", pad._rumble_ok, False)
    ps.ax[4] = -1.0                        # triggers back at rest
    hp = GamepadInput(joystick=ps, steer_limit=False, user_config=False,
                      mapping={"buttons": {5: "pause"}})
    check_eq("mapping= replaces the button table", hp.map["buttons"], {5: "pause"})

    if verbose:
        print("\n-- pause menu on the pad: OPTIONS opens, seeded edges, stick nav --")
    ps.bt = [False] * 16
    ps.ax[:4] = [0.0, 0.0, 0.0, 0.0]
    pad.poll_events()                      # settle the edge state
    ps.bt[6] = True                        # OPTIONS pressed ...
    check_eq("ps: options -> 'menu'", pad.poll_events(), ["menu"])
    pad.set_menu(True)                     # ... the sim opens the menu, still held
    check_eq("ps: the opening press is not a second edge", pad.poll_events(), [])
    ps.bt[6] = False
    pad.poll_events()
    ps.bt[12] = True
    check_eq("ps: d-pad down -> 'nav_down'", pad.poll_events(), ["nav_down"])
    ps.bt[12] = False
    ps.bt[0] = True
    check_eq("ps: cross -> 'select'", pad.poll_events(), ["select"])
    ps.bt[0] = False
    ps.bt[10] = True                       # R1 while the menu is open
    check_eq("ps: menu swallows R1 (no shift)",
             (pad.poll_events(), pad.update(DT, 10.0).gear_req), ([], 0))
    ps.bt[10] = False
    ps.ax[1] = -0.9
    check_eq("ps: stick up -> 'nav_up' once", pad.poll_events(), ["nav_up"])
    check_eq("ps: stick held -> nothing more", pad.poll_events(), [])
    ps.ax[1] = 0.0
    pad.poll_events()
    ps.bt[1] = True                        # circle = back, still held on exit
    check_eq("ps: circle -> 'back'", pad.poll_events(), ["back"])
    pad.set_menu(False)
    check_eq("ps: circle held across the close does not toggle the wing",
             pad.poll_events(), [])
    ps.bt[1] = False
    pad.poll_events()
    ps.bt[1] = True
    check_eq("ps: a fresh circle press is 'wing' again", pad.poll_events(), ["wing"])
    ps.bt[1] = False
    import io, contextlib
    ps0 = _StubPS()
    ps0.ax = [0.0] * 6                     # attach before the first HID report
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p0 = GamepadInput(joystick=ps0, steer_limit=False, user_config=False)
        c0 = p0.update(DT, 0.0)
    check_eq("rest check deferred while every axis reads 0.0 (no report yet)",
             ("rests at" in buf.getvalue(), p0._rest_checked), (False, False))
    # (a + 1) / 2 of an unreported axis is HALF TRAVEL: without the gate the
    # pad handed the car 47% throttle AND 47% brake on hot-plug.
    check_eq("no HID report yet -> both pedals exactly 0",
             (c0.throttle, c0.brake), (0.0, 0.0))
    ps0.ax[4] = ps0.ax[5] = -1.0           # the DualSense's first report (measured)
    with contextlib.redirect_stdout(buf):
        c0 = p0.update(DT, 0.0)
    check_eq("first report with triggers at -1: checked, no warning",
             ("rests at" in buf.getvalue(), p0._rest_checked), (False, True))
    check_eq("triggers resting at -1 -> pedals 0, rest latched at -1",
             (c0.throttle, c0.brake, p0._trig_rest["throttle"]), (0.0, 0.0, -1.0))
    ps0.ax[5] = 1.0
    check("R2 at the stop -> full throttle", p0.update(DT, 0.0).throttle,
          1.0, 1e-12)
    # the other convention in the wild: rest at 0.0, travel to +1.0. The old
    # (a+1)/2 read that as 50-100% of pedal; the latched rest makes it 0-100%.
    ps2 = _StubPS()
    ps2.ax = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ps2.ax[0] = 0.5                        # something non-zero so the rest check runs
    with contextlib.redirect_stdout(io.StringIO()):
        p2 = GamepadInput(joystick=ps2, steer_limit=False, user_config=False)
        c2 = p2.update(DT, 0.0)
    check_eq("a 0-resting trigger reads 0 at rest, not 0.47",
             (round(c2.throttle, 12), round(c2.brake, 12)), (0.0, 0.0))
    ps2.ax[5] = 1.0
    check("... and 1.0 at the stop", p2.update(DT, 0.0).throttle, 1.0, 1e-12)
    # the directional soft lock reaches the pad too: full stick INTO a slide
    # gets the floor, full stick OUT of it gets the catch.
    ps3 = _StubPS()
    ps3.ax = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0]
    with contextlib.redirect_stdout(io.StringIO()):
        p3 = GamepadInput(joystick=ps3, steer_limit=True,
                          k_us_deg=K_US_DEG_MEASURED, user_config=False)
    ps3.ax[0] = 1.0                        # full RIGHT stick = the catch (beta < 0)
    for _ in range(600):
        cp = p3.update(DT, 30.0, -12.0, 3000.0, 5)
    lim_out = steer_limit_pair_deg(30.0, -12.0, k_us_deg=K_US_DEG_MEASURED)[1]
    check("pad: full stick OUT of the slide reaches the catch limit",
          abs(math.degrees(cp.delta)), lim_out, 0.05, "deg")
    ps3.ax[0] = -1.0                       # full LEFT stick = into the slide
    for _ in range(2000):
        cp = p3.update(DT, 30.0, -12.0, 3000.0, 5)
    lim_into = steer_limit_pair_deg(30.0, -12.0, k_us_deg=K_US_DEG_MEASURED)[0]
    check("pad: full stick INTO the slide is held at the floor",
          abs(math.degrees(cp.delta)), lim_into, 0.05, "deg")
    ps1 = _StubPS()
    ps1.ax = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3]    # a mis-ordered driver: axis 5 mid-travel
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        GamepadInput(joystick=ps1, steer_limit=False, user_config=False)
    check_eq("trigger not at -1 once reported -> one-line warning",
             "rests at" in buf.getvalue(), True)
    bl = BlendedInput(KeyboardInput(steer_limit=True), pad, announce=False, hotplug=False)
    bl.set_menu(True)
    check_eq("blend: set_menu reaches both halves", (bl.kb.menu, pad.menu, bl.layout),
             (True, True, "ps"))
    bl.set_menu(False)

    if verbose:
        print("\n-- gearbox modes: auto / manual / clutch through the blend --")
    check_eq("default mode is automatic", bl.gearbox, "auto")
    for mode, flags in (("manual", (False, True)), ("clutch", (False, False)),
                        ("auto", (True, True))):
        bl.set_gearbox(mode)
        c = bl.update(DT, 10.0)
        check_eq(f"set_gearbox({mode!r}) -> Controls flags",
                 (c.auto_gearbox, c.auto_clutch, bl.gearbox), (*flags, mode))
    check_eq("gearbox_flags('nonsense') is automatic", gearbox_flags("nonsense"),
             (True, True))
    bl.set_steer_limit(False)
    check_eq("set_steer_limit reaches both halves",
             (bl.kb.steer_limit, pad.steer_limit), (False, False))
    bl.set_steer_limit(True)
    ps.bt[0] = True                         # cross held across a session boundary
    pad.seed_edges()
    check_eq("seed_edges: a held button is not a fresh edge", pad.poll_events(), [])
    ps.bt[0] = False
    pad.poll_events()
    ps.bt[0] = True
    check_eq("...but a new press after release is", "handbrake" in pad.map["buttons"].values()
             and pad.update(DT, 10.0).handbrake == 1.0, True)
    ps.bt[0] = False

    if verbose:
        print("\n-- ScriptedInput: no pygame, no wall clock, bit-identical --")

    def _script(t, veh, track):
        return Controls(delta=0.02 * math.sin(2.0 * t), throttle=min(1.0, t))

    def _run():
        si = ScriptedInput(_script)
        return [si.update(DT).delta for _ in range(2000)]

    a, b = _run(), _run()
    check_eq("two runs are bit-identical", a == b, True)
    check_eq("t = n_steps*dt, never accumulated",
             ScriptedInput(_script).update(DT).delta, 0.0)
    si = ScriptedInput(_script)
    for _ in range(1000):
        si.update(DT)
    check("t after 1000 steps", si.t, 1.0, 0.0, "s")
    check("scripted delta passes through unlimited",
          _script(0.4, None, None).delta, 0.02 * math.sin(0.8), 0.0, "rad")

    if verbose:
        print("\n-- against the real chassis: Controls fits, and RIGHT turns right --")
    from .vehicle import Vehicle          # already imported for Controls

    veh = Vehicle()
    veh.reset(0.0, 0.0, 0.0, V=20.0, gear=3)
    kb = KeyboardInput(steer_limit=True)
    kb.set_keys(**{k: False for k in kb.held})
    kb.set_keys(right=True, up=True)
    for _ in range(600):
        c = kb.update(DT, abs(veh.u), math.degrees(veh.beta), veh.rpm, veh.gear)
        veh.step(c, (1.0,) * 4, (1.0,) * 4, DT)
    check_eq("Controls is accepted by Vehicle.step unmodified", veh.n_steps if
             hasattr(veh, "n_steps") else veh.state.n_steps, 600)
    check_eq("holding RIGHT gives a right-hand (clockwise) yaw rate", veh.r < 0.0, True)
    check_eq("holding RIGHT gives a_y to the right", veh.ay < 0.0, True)
    check("delta commanded is the limiter's value", abs(math.degrees(c.delta)),
          kb.delta_lim_deg, 1e-9, "deg")

    if verbose:
        print()
        if ok:
            print("input.py: ALL CHECKS PASS")
        else:
            print(f"input.py: {len(fails)} FAILURE(S): " + ", ".join(fails))
    return ok


def calibration_audit(radii: Sequence[float] = (30.0, 50.0, 100.0, 130.0),
                      verbose: bool = True) -> dict:
    """Regenerate K_US_DEG_MEASURED from the chassis (harness.txt's own pitfall).

    Slow (~2.6 s per radius: each one is a converged ramp-steer solve in
    drive.vehicle), which is why it is not part of self_check(). Run it with
    `python3 -m drive.input --calibrate` whenever the chassis moves.
    """
    from .vehicle import steady_state_corner

    rows = []
    for R in radii:
        d = steady_state_corner(float(R))
        ack = math.degrees(L_WB / R)
        k_us = (d["delta_deg"] - ack) / d["ay_g"]
        lim_spec = steer_limit_deg(d["V"], 0.0, k_us_deg=K_US_DEG)
        lim_meas = steer_limit_deg(d["V"], 0.0, k_us_deg=K_US_DEG_MEASURED)
        rows.append(dict(R=R, V=d["V"], ay_g=d["ay_g"], delta_deg=d["delta_deg"],
                         ackermann_deg=ack, k_us_deg=k_us,
                         lim_k32=lim_spec, lim_meas=lim_meas,
                         margin_k32=lim_spec / d["delta_deg"] - 1.0,
                         margin_meas=lim_meas / d["delta_deg"] - 1.0))
    if verbose:
        print("steer-limiter calibration against drive.vehicle.steady_state_corner")
        print("    R      V     ay_g   delta   ack    k_us   lim(3.2) marg   "
              "lim(7.82) marg")
        for r in rows:
            print(f"  {r['R']:5.0f} {r['V']:6.3f} {r['ay_g']:6.4f} "
                  f"{r['delta_deg']:6.3f} {r['ackermann_deg']:5.3f} "
                  f"{r['k_us_deg']:6.3f} {r['lim_k32']:7.3f} "
                  f"{100*r['margin_k32']:+6.1f}% {r['lim_meas']:8.3f} "
                  f"{100*r['margin_meas']:+6.1f}%")
        print("  a negative margin means the driver aid, not grip, is the limit")
    return {"rows": rows,
            "k_us_deg_fit": sum(r["k_us_deg"] for r in rows) / len(rows)}


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        calibration_audit()
        sys.exit(0)
    if "--keys" in sys.argv:
        print(KEY_HELP)
        sys.exit(0)

    sys.exit(0 if self_check() else 1)
