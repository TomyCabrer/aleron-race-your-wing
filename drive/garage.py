"""3D garage: build the car's three wings -- a panel on each flank and a
wing on top -- design them with AeroBO, keep them in a library, and drive
exactly that car.

What changed from the one-panel editor
--------------------------------------
The published study has ONE flank panel (S 0.35 m^2, CL0 0.70 / 1.25, L/D
3.2) placed by four numbers. That car is still here, bit-for-bit, as the
built-in wings 'fin' and 'plate'. Around it:

* a BUILD has three slots -- left flank, right flank, top -- each holding
  a wing from the library at a station, a height and an incidence. Left
  and right mirror each other unless you unlock them.
* a WING is a WingSpec (drive/aero/wing.py). It is designed on two gated
  pages, AeroBO's order: the MISSION page (a lap of one of carsim's
  circuits, stated before anything is judged) and the DESIGN navigator --
  2 Airfoil / 2.8 Endplate / 3 Wing / 4 Results, AeroBO's stages. Since the
  AeroBO pivot (PLAN2) nothing on those pages is carsim's engine: every
  screen, section search, wing search and law is AeroBO's own, vendored
  unmodified at aerobo/ and driven through drive/aerobo_models.py on a
  worker thread. The mission stays carsim's and supplies AeroBO's operating
  point; the winning design flies in the game with AeroBO's forces (a law
  sampled from AeroBO's evaluator). The AIRFOIL page browses carsim's
  section library; the LIBRARY page saves and loads wings and whole builds,
  so a wing drawn for one car goes on the next. Every page takes the mouse
  as well as the keyboard.
* what the physics reads is small: an affine lift law, a quadratic drag
  law and two stall clamps per panel (vehicle.DevAero) and one downforce
  point for the top wing (vehicle.TopAero). The 1 kHz step never sees a
  solver.

Software-rendered as before (painter's, flat shaded, no OpenGL); the wing
meshes are true lofts of the chosen section.

Frame: body frame, x forward, y LEFT, z up, ground at z = 0, CG at
(0, 0, h_cg). CONTRACT section 0 with z added.

    python3 -m drive.drive --garage        # garage -> ENTER / cross -> drive
    python3 -m drive.garage                # self-check + screenshots
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field, fields, replace

import numpy as np
import pygame

from corsa_c import CorsaC, RHO, G
import crossover
from .menu import Menu, StickNav
from . import garage_ui as ui
from . import bodies
from . import design_jobs as dj
from .cae.form import Form
from .design_shell import CaeTree, DesignShell, SHELL_PAGES
from .aero.library import Library
from .aero.wing import WingSpec, BOUNDS, V_REF, design_point, re_bank_snap, wing_mass
from .aero import mission as ms
from .track import make_track
from .vehicle import DevAero, TopAero

CAR = CorsaC()

# --- published exterior dims (same numbers as render.py) -------------------
CAR_X_FRONT = 1.7515     # m ahead of the CG
CAR_X_REAR = -2.0675     # m behind the CG
CAR_HALF_W = 0.823       # m
CAR_H = 1.440            # m published height
WHEEL_R = 0.2915         # m 175/65R14
WHEEL_W = 0.175          # m
WHEEL_XY = ((0.9715, 0.7145), (0.9715, -0.7145),
            (-1.5195, 0.7100), (-1.5195, -0.7100))
WHEEL_Y_DRAW = 0.745     # m  centre pushed 30 mm out so the outer face clears
                         #    the sill; the physics track is untouched

# --- device geometry (render.py, est; the device is unpublished) -----------
S_DEV = 0.35             # m^2 ONE panel (ledger.S_DEV) -- the published panel
DEV_CHORD = 0.45         # m
DEV_SPAN = S_DEV / DEV_CHORD   # 0.78 m, vertical
DEV_THICK = 0.03         # m
DEV_OUT0 = 0.25          # m stowed standoff from the flank
DEV_OUT1 = 0.35          # m extra standoff when deployed
LD_DEV = 3.2
DCLDA = 2.47             # /rad  VehicleConfig.dCLda
CL_STALL = 1.6
CL0 = {"off": 0.0, "fin": 0.70, "plate": 1.25}
WING_TYPES = ("off", "fin", "plate")

# --- editable ranges ---------------------------------------------------------
X_W_MIN = CAR_X_REAR + 0.5 * DEV_CHORD      # the panel stays on the body
X_W_MAX = CAR_X_FRONT - 0.5 * DEV_CHORD
H_W_MIN = 0.40
H_W_MAX = 1.20
INC_MIN = -10.0
INC_MAX = 15.0
STEP_X = 0.05
STEP_H = 0.05
STEP_INC = 1.0
TOP_X_MIN = CAR_X_REAR + 0.15
TOP_X_MAX = 0.55
TOP_H_MAX = 1.85
TOP_STOW_GAP = 0.06      # m above the deck when stowed

V_R100 = 29.0875         # m/s  qss corner speed at R = 100 m, dry (README)
UNDERSTEER_MARGIN = 0.05
GAIN_CAP_PCT = 100.0 * (math.sqrt(1.0 + UNDERSTEER_MARGIN) - 1.0)

DESIGN_PATH = os.path.join("runs", "garage_design.json")
SLOTS = ("left", "right", "top")
#: the player's words for the slots. The flanks are the SIDE wings, as the G
#: modes call them (the owner, 2026-09-25: "Left flank and right flank, should
#: be side"), and the pair has ONE name, not a left and a right one (the owner,
#: 2026-09-26: "change right and left are given independent name (just side
#: wing)"): mirrored they are one design. The keys and the "flank" role stay
#: the code's own vocabulary.
SLOT_LABEL = {"left": "SIDE WING", "right": "SIDE WING", "top": "TOP WING"}
SLOT_ROLE = {"left": "flank", "right": "flank", "top": "top"}
#  task 45: what a player reads for the four built-in wings -- a name and
#  what the wing does, in one line -- keyed by their library names, which
#  stay as they are (every saved build, challenge and reference names them).
#  The lines are the car page's own numbers at 0 deg (`slot_summary`): the
#  plate +2.2 % corner speed for 69 N of drag, the fin +1.2 % for 39 N, the
#  E423 panel +0.9 % for 9 N and +2.1 % for 38 N at 15 deg. A wing of the
#  player's own is shown by its own name (`wing_shown`).
BUILTIN_WING_SHOWN = {
    "fin": ("Side fin", "a plain fin: some corner grip, some drag"),
    "plate": ("Side plate", "end plates: the most corner grip, the most drag"),
    "flank-e423": ("Low-drag side wing", "the least drag; more incidence ([ ]) adds grip"),
    "rear-s1223": ("Rear wing", "downforce, and drag that helps braking"),
}
#  The two DESIGN pages are walked in order -- mission, then the navigator --
#  and the order is enforced, not suggested (`Garage.open_section`). 'airfoil'
#  and 'library' are not steps: they browse what already exists; so do task
#  46's two card pages, 'wings' (SAVED WINGS, L) and 'cars' (SAVED CARS, G).
PAGES = ("car", "mission", "section", "airfoil", "library", "wings", "cars")
#  the two DESIGN pages, in order. 'section' is the navigator over everything
#  downstream of the mission -- airfoil, end plate, wing and results -- so the
#  old separate wing page is a GROUP of it now, not a page. Both are drawn
#  and driven by the AeroBO shell (`drive/design_shell.py`, SHELL_PAGES).
DESIGN_STEPS = ("mission", "section")
#  seconds of each frame the live runs may spend (`RunManager.pump`): the
#  first unit of a frame always runs, another only while it fits
RUN_BUDGET_S = 0.010

# the pause menu's help columns (ESC / OPTIONS). Task 45: they stack in one
# column right of the menu's rows, so with a pad connected both tables share
# its height -- 20 + 11 rows ran 15 px past the panel, off the bottom of the
# 1280x800 window. Rows that say one thing are one row now (C with the
# camera, the arrows together, the mouse with ESC; the pad's sticks and face
# buttons in pairs), and no text is longer than HELP_TEXT_MAX, so none wraps
# beside the menu's widest row (`_check_menu_fits` lays out the worst of
# them). T's 'on brake+steer' is the top wing's 'active' mode, as the car
# panel says it ('deploys  brake + steer (active)'). Task 46: K (save the
# slot's wing) took a row, so M and T share one -- a row more and the help
# beside the menu ran into its footer at 1280x720 with a pad
# (`_check_menu_fits`) -- and L / G are the two card pages (SHIFT+L the
# library as a list, named on those pages' bars).
GARAGE_HELP_KB = [
    ("mouse drag / wheel", "orbit / zoom  (C: reset camera)"),
    ("1 / 2 / 3, TAB", "select slot: left / right / top"),
    ("LEFT/RIGHT, UP/DOWN", "station x, height h; SHIFT: 1 cm"),
    ("[ / ]", "incidence -1 / +1 deg"),
    ("W / SHIFT+W", "next / previous library wing"),
    ("M / T", "mirror sides / top wing's mode"),
    ("SPACE", "deploy preview (0.45 s actuator)"),
    ("D", "design a wing (mission first)"),
    ("A / L / G", "airfoils / saved wings / cars"),
    ("K / SHIFT+K", "save the slot's wing / as new"),
    ("S / SHIFT+S", "save the build / as a new name"),
    ("B / SHIFT+B", "next / previous build (this car)"),
    ("F", "make it the car's default build"),
    ("R R / U", "all wings off / put them back"),
    ("ENTER", "drive this car"),
    ("H", "wing tutorial's box: hide / show"),
    ("ESC", "this menu  (mouse: click a row)"),
]
GARAGE_HELP_PAD = [
    ("left stick / d-pad", "move the wing (x, h) / step it"),
    ("right stick / R3", "orbit / reset camera"),
    ("L1 / R1", "incidence -1 / +1 deg"),
    ("TRIANGLE / SQUARE", "next slot / next library wing"),
    ("CIRCLE", "deploy preview"),
    ("L3", "design a wing for this slot"),
    ("CROSS", "drive this car"),
    ("OPTIONS", "this menu  (car, save, load)"),
]
#  the longest help text, in characters: at 8 px a character it fits beside
#  the menu's widest row (the build rows with MENU_NAME_MAX names, a running
#  wing tutorial's rows and their "v 16 more") without wrapping; at 1280x720
#  (13 px text) with 3 characters to spare, the tightest size laid out
HELP_TEXT_MAX = 32
#  a build's name in the pause menu's rows is cut to this many characters,
#  "..." ending a cut one ('my express' is whole): 24 characters made the
#  rows so wide that the help beside them wrapped line after line off the
#  screen. The car page's header and the library show the whole name.
MENU_NAME_MAX = 10
GARAGE_MENU_KEYS = {
    pygame.K_UP: "nav_up", pygame.K_DOWN: "nav_down",
    pygame.K_RETURN: "select", pygame.K_KP_ENTER: "select", pygame.K_SPACE: "select",
    pygame.K_ESCAPE: "menu", pygame.K_p: "menu",
    pygame.K_r: "defaults", pygame.K_c: "camera",
}
#  task 45: what cannot be taken back takes two presses. The first press of
#  R (all wings off) or DEL (a saved build or wing) only ARMS it and says what
#  a second press will do; the second, within ARM_S seconds and with no other
#  key between, does it. A held key's auto-repeat is never the second press.
ARM_S = 2.5
#  the keys that may confirm the pause menu's armed "Reset car" row: the
#  menu's select keys and its R hotkey (the pad's CROSS and a click select too)
MENU_CONFIRM_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_r)
#  ... and its armed "Quit to desktop" row: the select keys alone
MENU_SELECT_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
#  task 45: a garage hint shows HINT_S seconds from the moment it is set and
#  fades out over the last HINT_FADE_S of them (it used to stand until the
#  next one replaced it, however stale)
HINT_S = 4.0
HINT_FADE_S = 0.5
#  the pause menu's rows whose first select only arms: row action -> the arm
MENU_ARMS = {"defaults": "reset", "quit": "quit"}
PAD_NAMES_POLLED = ("r1", "l1", "up", "down", "right", "left", "triangle", "square",
                    "circle", "r3", "l3", "cross", "options")

# --- colours -------------------------------------------------------------------
C_BG = (27, 29, 33)
C_GROUND = (36, 38, 43)
C_GRID = (50, 53, 59)
C_SHADOW = (20, 21, 24)
C_PAINT = (215, 195, 74)
C_PAINT_DARK = (160, 146, 56)
C_GLASS = (62, 74, 92)
C_UNDER = (30, 32, 36)
C_TYRE = (52, 55, 60)
C_RIM = (168, 170, 176)
C_ARCH = (30, 32, 36)
C_PANEL_ON = (255, 140, 43)
C_PANEL_OFF = (107, 111, 117)
C_PANEL_SEL = (255, 196, 120)
C_STRUT = (90, 94, 100)
C_PLATE = (200, 110, 40)
C_DIM = (79, 163, 255)
C_TEXT = (232, 234, 238)
C_TEXT_DIM = (139, 144, 153)
C_WARN = (226, 82, 63)
C_OK = (78, 194, 106)
C_HUD_BG = (16, 17, 20)
C_VEC_F = (78, 194, 106)      # the wing's FORCE (side force / downforce) arrow
C_VEC_D = (226, 82, 63)       # its DRAG arrow
VEC_M_PER_N = 1.0 / 400.0     # one world scale for all three wings: 400 N = 1 m
FONT_NAMES = ("Menlo", "Monaco", "DejaVu Sans Mono", "Courier New")
LIGHT_DIR = np.array([0.45, 0.55, 0.70])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


# =========================================================================== #
#  THE PUBLISHED DESIGN (kept: the closed form and its four numbers)           #
# =========================================================================== #
@dataclass
class WingDesign:
    """The four numbers `vehicle.py` reads for the published panel, plus
    their closed-form readouts. Still the description of the study's car;
    `CarBuild` wraps it for the three-slot garage."""

    wing: str = "plate"
    x_w: float = 0.97
    h_w: float = 0.90
    inc_deg: float = 0.0

    # -- editing -----------------------------------------------------------
    def clamp(self) -> "WingDesign":
        self.x_w = min(max(self.x_w, X_W_MIN), X_W_MAX)
        self.h_w = min(max(self.h_w, H_W_MIN), H_W_MAX)
        self.inc_deg = min(max(self.inc_deg, INC_MIN), INC_MAX)
        if self.wing not in WING_TYPES:
            self.wing = "off"
        return self

    def cycle_type(self) -> None:
        self.wing = WING_TYPES[(WING_TYPES.index(self.wing) + 1) % len(WING_TYPES)]

    def reset(self) -> None:
        self.wing, self.x_w, self.h_w, self.inc_deg = "plate", 0.97, 0.90, 0.0

    # -- what the physics sees --------------------------------------------
    def cfg_kwargs(self) -> dict:
        return dict(wing=self.wing, x_w=round(self.x_w, 4), h_w=round(self.h_w, 4),
                    delta_dev_geom=math.radians(self.inc_deg))

    # -- closed form (crossover.py) -----------------------------------------
    def cl_eff(self) -> float:
        """CL0 + dCLda * incidence, clamped exactly as vehicle._aero clamps it."""
        cl0 = CL0[self.wing]
        if cl0 <= 0.0:
            return 0.0
        cl = cl0 + DCLDA * math.radians(self.inc_deg)
        return min(max(cl, 0.0), CL_STALL)

    def k(self) -> float:
        return 0.5 * RHO * S_DEV * self.cl_eff()

    def multiplier(self) -> float:
        return (self.x_w + CAR.b) / CAR.b

    def force_N(self, V: float = V_R100) -> float:
        return 0.5 * RHO * V * V * S_DEV * self.cl_eff()

    def gain_pct(self, R: float) -> float | None:
        g = crossover.gain(self.k(), R, self.x_w)
        return None if g is None else 100.0 * g

    def station_label(self) -> str:
        return station_label(self.x_w)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str = DESIGN_PATH) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str = DESIGN_PATH) -> "WingDesign | None":
        try:
            with open(path) as f:
                d = json.load(f)
            if int(d.get("version", 1)) >= 2:
                return None                    # a CarBuild file: see CarBuild.load
            return cls(wing=str(d.get("wing", "plate")), x_w=float(d.get("x_w", 0.97)),
                       h_w=float(d.get("h_w", 0.90)),
                       inc_deg=float(d.get("inc_deg", 0.0))).clamp()
        except (OSError, ValueError, TypeError):
            return None


#: task 45: how far either side of an axle a station still reads as AT those
#: wheels, m
AXLE_BAND = 0.3


def station_label(x: float, car=None) -> str:
    """Where station `x` sits on `car` (a `cars.py` key or spec, None the
    Corsa), in words that hold for any body: by the car's OWN axles, read
    off `bodies.body(car)`, with AXLE_BAND either side of each. Task 45:
    the labels were the Corsa's parts at the Corsa's stations, so a flank
    on the two-door MX-5 sat at the 'rear door' and one mid-bus at the
    'front bumper'."""
    b = bodies.body(car)
    x_f, x_r = b.a, b.a - b.L             # the front and rear axles
    if x > x_f + AXLE_BAND:
        return "ahead of the front wheels"
    if x >= x_f - AXLE_BAND:
        return "at the front wheels"
    if x > x_r + AXLE_BAND:
        return "between the wheels"
    if x >= x_r - AXLE_BAND:
        return "at the rear wheels"
    return "behind the rear wheels"


def split_text(x: float, car) -> tuple:
    """A top wing's downforce split between the axles at station `x` on
    `car` (a `cars.py` spec): (the words the pages print, the front share
    clamped to 0..1). Task 45: a wing behind the rear axle read 'front -3%
    rear 103%'. Past an axle every newton lands on that axle's wheels (and
    the lever lightens the other end), so the words say so instead of a
    share outside 0-100 %."""
    share = (x + car.b) / car.L
    if share < 0.0:
        return "all on the rear (behind the rear axle)", 0.0
    if share > 1.0:
        return "all on the front (ahead of the front axle)", 1.0
    return f"front {100 * share:.0f}%  rear {100 * (1 - share):.0f}%", share


def shown_spec(spec: WingSpec) -> WingSpec:
    """`spec` as the car page reads it (task 45): a published panel the
    library has not analysed yet -- the 'fin' a first W fits -- gets a
    display-only aero (RHO and its role's V_REF), so `design_point` reads its
    closed form and the page shows its numbers and arrows like any other
    wing's. A copy: the library's own spec is left as it is."""
    if spec.aero or not spec.legacy:
        return spec
    return replace(spec, aero=dict(rho=RHO, V_ref=V_REF[spec.role]))


def wing_shown(name: str, lib=None) -> tuple[str, str]:
    """(the name a player reads, what the wing does) for library wing `name`
    (task 45): a built-in's pair from BUILTIN_WING_SHOWN, any other wing its
    own name and no line, an empty slot 'no wing'. With `lib`, a wing of the
    table's name that is not the library's built-in is the player's own."""
    if not name:
        return "no wing", ""
    w = lib.wings.get(name) if lib is not None else None
    if name in BUILTIN_WING_SHOWN and (w is None or w.builtin):
        return BUILTIN_WING_SHOWN[name]
    return name, ""


def slot_summary(spec: WingSpec, key: str, dp: dict) -> tuple:
    """The car page's first lines for fitted slot `key` (task 45): what the
    wing does for the car, in a player's words -- (the numbers, what they
    are taken at, their colour). A flank panel's corner-speed gain in a
    100 m corner (`design_point`'s own, crossover.gain at R = 100 m) or a
    top wing's downforce, the drag at the role's reference speed, and the
    weight the car carries for it (`wing_mass` at the standoff
    `CarBuild.mass_points` charges). Red for a stalled wing, and for a
    flank panel that loses corner speed or passes the understeer cap."""
    kmh = f"{3.6 * dp['V']:.0f} km/h"
    flank = SLOT_ROLE[key] == "flank"
    kg = wing_mass(spec, DEV_OUT0 + DEV_OUT1 if flank else DEV_OUT0)
    if flank:
        g = dp.get("gain_pct")
        what = "corner speed off the scale" if g is None else f"corner speed {g:+.1f} %"
        col = C_OK if g is not None and 0.0 < g <= GAIN_CAP_PCT else C_WARN
        why = f"in a 100 m corner; drag at {kmh}"
    else:
        what, col = f"downforce {dp['F']:.0f} N", C_OK
        why = f"downforce and drag at {kmh}"
    if dp.get("stalled"):
        col = C_WARN
    return f"{what} · drag {dp['D']:.0f} N · +{kg:.1f} kg", why, col


# =========================================================================== #
#  THE BUILD: three slots                                                      #
# =========================================================================== #
@dataclass
class Slot:
    wing: str = ""            # library wing name, "" = nothing in this slot
    x: float = 0.97           # m, station (forward of the CG)
    h: float = 0.90           # m, height of the wing's centre above the ground
    inc_deg: float = 0.0      # built-in incidence
    mode: str = "active"      # top wing only: 'fixed' | 'active'


def deck_z(x: float) -> float:
    """The car's top surface height at station x (from the mesh stations)."""
    xs = [s[0] for s in STATIONS][::-1]
    zs = [s[3] for s in STATIONS][::-1]
    return float(np.interp(x, xs, zs))


#  ---- the car a build is FITTED to (task 41) --------------------------------
#  Until task 41 every slot band below was the Corsa's, whichever car was
#  driven. The three STOCK cars (Corsa, MX-5, 540i) still share exactly those
#  bands -- flank x from the Corsa's bumpers, flank h 0.40-1.20, top x to
#  0.55, top h from this garage's own Corsa deck (STATIONS below; the
#  renderer's re-cut hatch tail sits up to 0.17 m lower) + 0.14 to 1.85 --
#  so no build saved before task 41 moves on any of them (`bodies.
#  STOCK_STYLES`; bodies carries a copy of this deck, `LEGACY_DECK`, and this
#  module's self-check proves the copy). Only their span LIMITS are per car.
#  A new car (the Express, the bus) reads its bands off its own body shell
#  (`drive/bodies.py`).
def _fit_car(build, car):
    """The car `build` is clamped to: `car` when given (a `cars.py` key or a
    CarSpec), else the car the build says it was made for (`build.car`,
    when the build carries one), else the Corsa (None)."""
    if car is not None:
        return car
    return getattr(build, "car", "") or None


#: task 46: how far inside the tail a top wing's structure stands at the least
TOP_FOOT_MARGIN = 0.02


def top_x_floor(car, spec: "WingSpec | None") -> float:
    """The rearmost station a top wing may take on `car` so that what holds
    it stands ON the car (task 46, the owner: "Make it so that the wing
    placed both pylons (or in endplate pylons case the endplate) are both
    connected to a physical part of the car"): its swan necks' feet, a
    gap and a pylon's width behind the trailing edge (`aero.wing.
    pylon_rings`), or its carrying plates' chord, TOP_FOOT_MARGIN inside
    the tail. At the band's old floor, 0.15 m inside the tail, a 0.30 m
    wing's pylons stood 6 cm behind the car, in the air. A wing with no
    mount ('none') keeps the band."""
    from .aero.wing import PYLON_TE_GAP, PYLON_W, plate_chord_scale
    x_rear = bodies.body(car).x_rear
    if spec is None or spec.mount not in ("pylon", "endplate"):
        return -1e9
    if spec.mount == "pylon":
        back = 0.5 * float(spec.chord) + PYLON_TE_GAP + PYLON_W
    else:                          # the plates' own chord, at the tip (their widest)
        back = 0.5 * float(spec.chord) * max(1.0, plate_chord_scale(
            "top", spec.plate_chord_ratio, bool(spec.plate_chord_follows)))
    return x_rear + back + TOP_FOOT_MARGIN


def top_h_band(car, x: float) -> tuple[float, float]:
    """The top wing's height band at station x on `car`: clear of THAT car's
    deck by the stowed gap + 0.08 m, up to 0.41 m over its roof
    (`bodies.top_h_band`; a Citaro's top wing rides ~3.3 m up). A stock car:
    `deck_z(x) + TOP_STOW_GAP + 0.08` to 1.85, the garage's rule unchanged."""
    return bodies.top_h_band(car, x)


def flank_h_floor(car, span: float, unlimited: bool = False) -> float:
    """The lowest mount height a flank panel of `span` may take on `car` in
    Real mode: its lower tip at the car's ground clearance, h = ground +
    span / 2 (`bodies.span_limit`'s rule, solved for h). In Unlimited mode
    (task 45) the tip may go past the clearance but not into the road: its
    lower tip at 0 m, h = span / 2."""
    return (0.0 if unlimited else bodies.body(car).ground) + 0.5 * float(span)


@dataclass
class CarBuild:
    """Three slots, a name, and the mirror lock. `cfg_kwargs(lib)` turns it
    into VehicleConfig fields; `hud_kwargs(lib)` into what the renderer
    draws. A build whose flanks carry the SAME published panel and no top
    wing maps onto the closed-form path exactly (that is the study's car)."""

    name: str = "my civetta"      # = default_build_name("corsa")
    left: Slot = field(default_factory=lambda: Slot("", 0.97, 0.90, 0.0))
    right: Slot = field(default_factory=lambda: Slot("", 0.97, 0.90, 0.0))
    top: Slot = field(default_factory=lambda: Slot("", -0.90, 1.55, 6.0, "active"))
    mirror: bool = True
    builtin: bool = False
    #: task 41: the car this build was made for, a `cars.py` key. "" is a
    #: build saved before builds knew their car: made for ANY car, and still
    #: offered to every one. The garage stamps its own car on every save; a
    #: LABEL, like the name -- `records.BUILD_META` strips it wherever two
    #: builds are compared or a PB is filed, so tagging a build moves no
    #: record. It decides only what a car is OFFERED: its own builds first,
    #: and never another car's build without the player choosing it.
    car: str = ""

    def slot(self, key: str) -> Slot:
        return getattr(self, key)

    # -- geometry limits -----------------------------------------------------
    def clamp(self, lib: "Library | None" = None, car=None) -> "CarBuild":
        """Hold every slot inside the bands of the car it is FITTED to (task
        41: `car`, a `cars.py` key or a CarSpec; None = `_fit_car`'s rule).
        The SPAN is not clamped here: a wing past its car's physical limit is
        kept, and whether the build is then an Unlimited one is a question
        for `bodies.over_limits`, asked at each run's start."""
        car = _fit_car(self, car)
        h_lo, h_hi = bodies.flank_h_band(car)
        for key in ("left", "right"):
            s = self.slot(key)
            c = DEV_CHORD
            if lib is not None and s.wing in lib.wings:
                c = lib.wings[s.wing].chord
            x_lo, x_hi = bodies.flank_x_band(car, c)
            s.x = min(max(s.x, x_lo), x_hi)
            s.h = min(max(s.h, h_lo), h_hi)
            s.inc_deg = min(max(s.inc_deg, INC_MIN), INC_MAX)
            s.mode = "active"
        t = self.top
        x_lo, x_hi = bodies.top_x_band(car)
        if lib is not None and t.wing in lib.wings:
            x_lo = min(max(x_lo, top_x_floor(car, lib.wings[t.wing])), x_hi)
        t.x = min(max(t.x, x_lo), x_hi)
        t_lo, t_hi = top_h_band(car, t.x)
        t.h = min(max(t.h, t_lo), t_hi)
        t.inc_deg = min(max(t.inc_deg, BOUNDS["top"]["inc_deg"][0]), BOUNDS["top"]["inc_deg"][1])
        t.mode = "active" if t.mode == "active" else "fixed"
        if lib is not None:
            for key in SLOTS:
                s = self.slot(key)
                w = lib.wings.get(s.wing)
                if s.wing and (w is None or w.role != SLOT_ROLE[key]):
                    s.wing = ""
        if self.mirror:
            self.right = Slot(self.left.wing, self.left.x, self.left.h, self.left.inc_deg, "active")
        return self

    def sync_mirror(self, edited: str) -> None:
        if not self.mirror or edited == "top":
            return
        src = self.slot(edited)
        dst = "right" if edited == "left" else "left"
        setattr(self, dst, Slot(src.wing, src.x, src.h, src.inc_deg, "active"))

    def wings(self, lib: "Library") -> dict:
        return {k: lib.wings.get(self.slot(k).wing) for k in SLOTS}

    def has_any(self, lib: "Library") -> bool:
        return any(w is not None for w in self.wings(lib).values())

    def reset(self, car=None) -> None:
        """No wings, every slot at `car`'s default station (task 41:
        `bodies.slot_defaults`; the stock cars' are the old (0.97, 0.90) /
        (-0.90, 1.55, 6 deg), a Citaro's top wing sits 3.30 m up)."""
        d = bodies.slot_defaults(_fit_car(self, car))
        (fx, fh), (tx, th, ti) = d["flank"], d["top"]
        self.left = Slot("", fx, fh, 0.0)
        self.right = Slot("", fx, fh, 0.0)
        self.top = Slot("", tx, th, ti, "active")
        self.mirror = True

    @classmethod
    def for_car(cls, car=None) -> "CarBuild":
        """The empty car for `car`: no wings, its own default slots."""
        b = cls()
        b.reset(car)
        return b

    # -- what the physics sees ---------------------------------------------
    def cfg_kwargs(self, lib: "Library") -> dict:
        w = self.wings(lib)
        wl, wr, wt = w["left"], w["right"], w["top"]
        L, R, T = self.left, self.right, self.top
        pure_legacy = (wt is None and (wl is None or wl.legacy) and (wr is None or wr.legacy)
                       and (wl is wr or (wl is None and wr is None))
                       and abs(L.x - R.x) < 1e-9 and abs(L.h - R.h) < 1e-9
                       and abs(L.inc_deg - R.inc_deg) < 1e-9)
        if pure_legacy:
            name = wl.name if wl is not None else "off"
            if name not in WING_TYPES:
                name = "off"
            return dict(wing=name, x_w=round(L.x, 4), h_w=round(L.h, 4),
                        delta_dev_geom=math.radians(L.inc_deg))
        kw = dict(wing="off", x_w=round(L.x, 4), h_w=round(L.h, 4),
                  delta_dev_geom=math.radians(L.inc_deg))
        for key, slot, spec in (("dev_left", L, wl), ("dev_right", R, wr)):
            kw[key] = _dev_aero(spec, slot)
        kw["top"] = _top_aero(wt, T, lib)
        return kw

    def mass_points(self, lib: "Library"):
        """The three fitted wings as `cars.PointMass`, at their own stations.

        This closes the loop task 3 opened: `aero.wing.wing_mass(spec)` is a
        bottom-up floor for a designed wing and its mount (two skins, two tip
        plates, two pylons of the mount standoff -- no ribs, no fasteners, no
        body reinforcement), and the natural consumer of that number is the
        car it is bolted to. A flank panel's mass acts at its slot `(x, h)`;
        the top wing's at its own, which is 1.5 m up and behind the rear axle
        and therefore the one that actually moves anything.

        An empty slot contributes nothing, so the DEFAULT car -- no wings --
        is untouched, which is why this is safe to charge at all: `cars.
        with_masses` returns the same object when the total is zero.

        Both flanks are charged. A mirrored build carries TWO panels, and
        pretending it carries one to keep the numbers tidy would be exactly
        the kind of stale book-keeping this batch exists to remove.

        Measured on the seeded library: `flank-e423` 4.91 kg a side (the
        published `fin` 4.85), `rear-s1223` 4.37 kg. A car with all three is
        1024.18 kg against 1010, `wdist_f` 0.6122 against 0.6100, `h_cg`
        0.5577 against 0.5500 -- the top wing at h = 1.57 m raises the CG by
        7.7 mm on its own -- and `Izz` 1212.7 against 1200. Small, and not
        nothing: 7.7 mm of CG height is 0.9 % more lateral load transfer, on
        a car whose whole device is worth 2.35 %.
        """
        import cars
        out = []
        for key in SLOTS:
            slot = self.slot(key)
            spec = lib.wings.get(slot.wing)
            if spec is None:
                continue
            # the same standoff the aero is analysed at (`_analyse` above):
            # a deployed flank panel stands off DEV_OUT0 + DEV_OUT1, the top
            # wing is carried on its own mount
            stand = DEV_OUT0 + DEV_OUT1 if SLOT_ROLE[key] == "flank" else DEV_OUT0
            out.append(cars.PointMass(float(wing_mass(spec, stand)),
                                      round(slot.x, 4), round(slot.h, 4),
                                      f"{key} wing {spec.name}"))
        return tuple(out)

    def mission_aero(self, lib: "Library", exclude: str | None = None) -> "ms.MissionAero":
        """This build's devices as `aero.mission` reads them, optionally with
        one slot left EMPTY.

        `exclude` is what makes the design pages honest: a wing is scored
        against the lap of the car it is going onto, with everything ELSE
        still fitted, so its own contribution is the difference and not a
        number confounded with its neighbours'. Excluding either flank
        excludes the flank device, because only ONE panel is deployed at a
        time (`vehicle.VehicleState.dev_side`) and a mirrored build's two
        panels are one device to the lap.

        Routed through `design_point`, which already knows how to read a
        LEGACY wing's published closed form as well as a designed wing's
        lattice fit -- so the study's own 'fin' and 'plate' fly the mission
        without a second code path.
        """
        out = ms.MissionAero()
        ex_flank = exclude in ("left", "right")
        if not ex_flank:
            slot = self.left if self.left.wing else self.right
            spec = lib.wings.get(slot.wing)
            if spec is not None:
                lib.analyse_wing(spec)
                dp = design_point(spec, slot.inc_deg, V=V_REF["flank"], x_w=slot.x)
                if dp and dp["D"] > 0.0 and dp["V"] > 0.0:
                    out = ms.MissionAero(k_dev=dp["F"] / (dp["V"] ** 2),
                                         ld_dev=dp["LD"], x_w=slot.x, h_w=slot.h)
        if exclude != "top":
            slot = self.top
            spec = lib.wings.get(slot.wing)
            if spec is not None:
                lib.analyse_wing(spec, ride_h=slot.h)
                dp = design_point(spec, slot.inc_deg, V=V_REF["top"], x_w=slot.x)
                if dp and dp["q"] > 0.0:
                    out = ms.MissionAero(k_dev=out.k_dev, ld_dev=out.ld_dev,
                                         x_w=out.x_w, h_w=out.h_w,
                                         cz_a=dp["F"] / dp["q"], cd_a=dp["D"] / dp["q"],
                                         cd_a_stowed=0.0, x_t=slot.x,
                                         top_mode=("active" if slot.mode == "active" else "fixed"))
        return out

    def hud_kwargs(self, lib: "Library") -> dict:
        w = self.wings(lib)
        wl, wr, wt = w["left"], w["right"], w["top"]
        ref = wl or wr
        out = dict(dev_left=wl is not None, dev_right=wr is not None,
                   x_w_left=round(self.left.x, 4), x_w_right=round(self.right.x, 4),
                   dev_chord=(ref.chord if ref is not None else DEV_CHORD),
                   dev_span=(ref.span if ref is not None else DEV_SPAN),
                   dev_plate=(ref.plate_h_flown if ref is not None else 0.0),
                   dev_mount=(ref.mount if ref is not None else "none"),
                   wing_left_name=(wl.name if wl is not None else ""),
                   wing_right_name=(wr.name if wr is not None else ""),
                   top_on=wt is not None, top_x=round(self.top.x, 4),
                   top_span=(wt.span if wt is not None else 0.0),
                   top_chord=(wt.chord if wt is not None else 0.0),
                   top_plate=(wt.plate_h_flown if wt is not None else 0.0),
                   top_mount=(wt.mount if wt is not None else "none"),
                   top_mode=self.top.mode,
                   wing_top_name=(wt.name if wt is not None else ""))
        #  how the plates and the pylons are DRAWN (`wing_polys`; the chase
        #  view's `wing_mesh3` reads the same numbers): taper for a chord law
        #  a plate may continue, the pylons' station, the plates' lean, blend,
        #  chord ratio and chord law. An empty slot keeps render's defaults,
        #  which are the WingSpec's -- a wing drawn as carsim always drew it
        for pre, spec in (("dev", ref), ("top", wt)):
            if spec is None:
                continue
            out.update({f"{pre}_taper": float(spec.taper),
                        f"{pre}_pylon_frac": float(spec.pylon_frac),
                        f"{pre}_plate_cant": float(spec.plate_cant_deg),
                        f"{pre}_plate_blend": float(spec.plate_blend),
                        f"{pre}_plate_shape": str(spec.plate_shape),
                        f"{pre}_plate_ratio": float(spec.plate_chord_ratio),
                        f"{pre}_plate_follows": bool(spec.plate_chord_follows),
                        #  AeroBO's own plate arc, bracketed to the body where
                        #  it stops short; 0 = a carsim plate, drawn to the body
                        f"{pre}_plate_own": (float(spec.plate_h) if spec.engine == "aerobo"
                                             else 0.0)})
        #  a flank its WingLab endplates carry deploys to its flown standoff
        #  (`flank_out`; stowed it retracts against the side); 0 = carsim's
        if carried_by_own_plates(ref):
            out["dev_standoff"] = float(ref.ride_h_flown)
        return out

    def summary(self, lib: "Library") -> str:
        """One line of the three slots, for a player: a built-in wing by its
        player name (task 45: `wing_shown`, as W's hint and the library)."""
        w = self.wings(lib)
        parts = []
        for k in SLOTS:
            if k == "right" and self.mirror:
                continue                                # one side wing: the pair mirrored
            s = self.slot(k)
            name = "top" if k == "top" else ("side" if self.mirror else f"side {k}")
            parts.append(f"{name}: {wing_shown(w[k].name, lib)[0] if w[k] else 'none'} "
                         f"x {s.x:+.2f} h {s.h:.2f} inc {s.inc_deg:+.0f}")
        return "   ".join(parts)

    # -- persistence ---------------------------------------------------------
    def to_json(self) -> dict:
        #  still version 2: "car" is one more label, and a version-2 file
        #  without it (every build saved before task 41) reads as car ""
        return dict(version=2, name=self.name, mirror=self.mirror, builtin=self.builtin,
                    car=self.car, slots={k: asdict(self.slot(k)) for k in SLOTS})

    @classmethod
    def from_json(cls, d: dict) -> "CarBuild":
        if not isinstance(d, dict):
            return cls()                        # a wrong-shaped file: the empty car
        if int(d.get("version", 1)) < 2 and "slots" not in d:
            # a WingDesign file: the published panel on both flanks
            wd = WingDesign(wing=str(d.get("wing", "plate")), x_w=float(d.get("x_w", 0.97)),
                            h_w=float(d.get("h_w", 0.90)), inc_deg=float(d.get("inc_deg", 0.0))).clamp()
            name = wd.wing if wd.wing != "off" else ""
            b = cls(left=Slot(name, wd.x_w, wd.h_w, wd.inc_deg), right=Slot(name, wd.x_w, wd.h_w, wd.inc_deg))
            return b
        sl = d.get("slots", {})
        if not isinstance(sl, dict):
            sl = {}

        def _slot(k, default):
            v = sl.get(k, {})
            if not isinstance(v, dict):
                v = {}
            try:
                return Slot(str(v.get("wing", default.wing)), float(v.get("x", default.x)),
                            float(v.get("h", default.h)), float(v.get("inc_deg", default.inc_deg)),
                            str(v.get("mode", default.mode)))
            except (TypeError, ValueError):        # a slot that is not numbers: the empty slot
                return default

        base = cls()
        import cars as _cars
        #  a hand-edited tag that is not a string: any car; a retired car's
        #  ('mx5', 'bus', task 46): the Corsa's (`cars.build_car_key`)
        car = _cars.build_car_key(d.get("car", ""))
        return cls(name=str(d.get("name", default_build_name(car or "corsa"))),
                   left=_slot("left", base.left),
                   right=_slot("right", base.right), top=_slot("top", base.top),
                   mirror=bool(d.get("mirror", True)), builtin=bool(d.get("builtin", False)),
                   car=car)

    def save(self, path: str = DESIGN_PATH) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"                     # a crash mid-write keeps the last car whole
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str = DESIGN_PATH) -> "CarBuild | None":
        """The last car built, or None. A file that is there but is not a
        build is moved aside to `<path>.bad` (kept, not deleted), so the
        garage opens and the next save writes a clean one."""
        try:
            with open(path) as f:
                return cls.from_json(json.load(f))
        except OSError:
            return None
        except (ValueError, TypeError, KeyError, AttributeError):
            try:
                os.replace(path, path + ".bad")
            except OSError:
                pass
            return None

    def copy(self) -> "CarBuild":
        return CarBuild.from_json(self.to_json())


def default_build_name(car: str = "corsa") -> str:
    """A new build's name on `car` (task 41): 'my civetta', 'my halcón',
    'my n540', 'my courier' -- the car's shown short tag
    (`prerace.car_label`), lower case, short enough for a list row. Before
    the fictional names it was the KEY ('my corsa'); an old save's build so
    called keeps that name, as any name the player chose."""
    from .prerace import car_label
    return f"my {(car_label(car or 'corsa') or 'car').lower()}"


def other_car_name(build: "CarBuild", car: str, lib: "Library | None" = None) -> bool:
    """Is `build` still called by ANOTHER car's new-build name -- 'my corsa'
    on the Express -- and not the library's build of that name? Task 45: a
    first launch makes its car from the Corsa's defaults ('my corsa', made
    for any car), and a player who then picked the Express in Settings found
    the garage's build called 'my corsa', and S offering to save it so. The
    garage calls such a build by its own car's name (`Garage._fit_in`). A
    name the player chose, or the library build loaded by that name (the
    Corsa's saved 'my corsa', same content), is kept."""
    name = str(getattr(build, "name", "") or "").strip()
    if not car or name == default_build_name(car):
        return False
    import cars as _cars
    keys = set(bodies.STYLE_OF) | set(_cars.CARS)
    #  task 49: the new-build names from before the fictional names ('my
    #  corsa', 'my express': the KEY) still count as a new-build name, so an
    #  old save's first-launch build is renamed to this car's, not shown
    if name not in {default_build_name(k) for k in keys} | {f"my {k}" for k in keys}:
        return False
    held = lib.builds.get(name) if lib is not None else None
    if isinstance(held, dict):
        from .prerace import _same_build
        return not _same_build(held, build.to_json())
    return True


def new_build(car: str = "corsa") -> CarBuild:
    """The EMPTY build for `car` (task 41): no wings, the car's own default
    slots (`bodies.slot_defaults`; the three stock cars' are `CarBuild()`'s
    own numbers), named `default_build_name(car)` and tagged with the car.
    What a car starts with when it has no default and the build in hand was
    made for another car."""
    b = CarBuild.for_car(car or "corsa")   # the car's own slots (part A's reset)
    b.name, b.car = default_build_name(car), str(car or "")
    return b


def _menu_name(name: str) -> str:
    """A build's name as the pause menu's rows show it: whole up to
    MENU_NAME_MAX characters, else cut to that length ending in '...'."""
    name = str(name or "")
    return name if len(name) <= MENU_NAME_MAX else name[:MENU_NAME_MAX - 3] + "..."


def _could_not_save(exc: Exception) -> str:
    """The hint for a library / file write that failed (a full disk, a
    read-only runs/, a name whose file another record holds): the garage
    stays up and says so."""
    return f"could not save: {getattr(exc, 'strerror', None) or exc}"


def _dev_aero(spec: "WingSpec | None", slot: Slot):
    """A flank slot's wing as `vehicle.DevAero`: the published closed form,
    or the wing's own law. An AeroBO wing's law is the one sampled from
    AeroBO's evaluator at its winning design (drive/aerobo_models), read as
    stored -- the flank flies with ground effect off, so the slot's x and h
    change nothing about it."""
    if spec is None:
        return None
    if spec.legacy:
        lg = spec.legacy
        return DevAero(name=spec.name, S=float(lg.get("S", S_DEV)), CL0=float(lg["CL0"]),
                       CLa=DCLDA, CL_min=0.0, CL_max=CL_STALL, cd0=0.0,
                       cd1=1.0 / float(lg.get("LD", LD_DEV)), cd2=0.0,
                       x_w=round(slot.x, 4), h_w=round(slot.h, 4), inc=math.radians(slot.inc_deg))
    if "CLa" not in spec.aero:
        return None
    return DevAero.from_aero(spec.aero, round(slot.x, 4), round(slot.h, 4), slot.inc_deg, spec.name)


def _top_aero(spec: "WingSpec | None", slot: Slot, lib: "Library"):
    """The top slot's wing as `vehicle.TopAero` at the slot's incidence.

    A carsim wing is re-analysed by carsim's lattice when the slot's height
    is not the ride height it was analysed at. An AeroBO wing never is: its
    `aero` IS AeroBO's law (`Library.analyse_wing` returns it as stored), and
    a slot moved since it was derived is re-derived through AeroBO before
    the build leaves the garage (`Garage.rederive_stale`), not here."""
    if spec is None:
        return None
    if spec.engine != "aerobo" and ("CLa" not in spec.aero
                                    or abs(float(spec.aero.get("ride_h") or 0.0) - slot.h) > 1e-6):
        lib.analyse_wing(spec, ride_h=slot.h)
    if "CLa" not in spec.aero:
        return None
    return TopAero.from_aero(spec.aero, slot.inc_deg, round(slot.x, 4), round(slot.h, 4),
                             slot.mode, spec.name)


# =========================================================================== #
#  THE MESH                                                                    #
# =========================================================================== #
# Cross-sections down the car: (x, z_floor, z_belt, z_top, half_w, half_w_roof)
STATIONS = (
    (CAR_X_FRONT, 0.22, 0.60, 0.66, 0.70, 0.52),   # bumper face
    (1.55, 0.17, 0.70, 0.76, 0.79, 0.62),
    (1.10, 0.15, 0.78, 0.83, CAR_HALF_W, 0.68),
    (0.60, 0.15, 0.85, 0.90, CAR_HALF_W, 0.70),    # scuttle
    (-0.05, 0.15, 0.88, 1.38, CAR_HALF_W, 0.64),   # A-pillar top
    (-0.80, 0.15, 0.90, CAR_H, CAR_HALF_W, 0.64),  # roof
    (-1.45, 0.15, 0.92, 1.40, CAR_HALF_W, 0.62),   # C-pillar
    (-1.80, 0.18, 0.95, 1.16, 0.80, 0.58),         # tailgate glass base
    (CAR_X_REAR, 0.28, 0.75, 0.86, 0.70, 0.50),    # rear bumper face
)
N_RING = 10
X_WINDSCREEN = (0.60, -0.05)      # roof segment between these = glass
X_REAR_GLASS = (-1.45, -1.80)
X_SIDE_GLASS = (-0.05, -1.45)     # belt->roof segment between these = glass


def _ring(st):
    x, zb, zbelt, ztop, w, wr = st
    return np.array([
        (x, -0.92 * w, zb), (x, -w, zb + 0.28), (x, -w, zbelt), (x, -wr, ztop),
        (x, 0.0, ztop + 0.02),
        (x, wr, ztop), (x, w, zbelt), (x, w, zb + 0.28), (x, 0.92 * w, zb),
        (x, 0.0, zb - 0.02),
    ])


def _orient(verts, inside):
    """Wind the polygon so its normal points AWAY from `inside`."""
    v = np.asarray(verts, dtype=np.float64)
    n = np.cross(v[1] - v[0], v[2] - v[0])
    if np.dot(n, v.mean(axis=0) - np.asarray(inside)) < 0.0:
        v = v[::-1].copy()
    return v


def _between(x0, x1, lo_hi):
    lo, hi = max(lo_hi), min(lo_hi)
    return (x0 <= lo + 1e-9 and x1 >= hi - 1e-9)


def build_car_mesh(paint=C_PAINT, car=None) -> list[tuple[np.ndarray, tuple, str]]:
    """(verts (n,3), colour, kind). Kind is 'body' | 'wheel' | 'trim'.

    `paint` is the body colour (the player's paint, drive/paint.py); the nose
    and tail caps are a darker tone of it. The stock yellow keeps its own
    hand-picked cap tone, C_PAINT_DARK; any other paint gets 0.75 of itself,
    about what C_PAINT_DARK is of C_PAINT (0.744-0.757 by channel).

    `car` (task 41): the car being fitted, a `cars.py` key. None and
    'corsa' are the garage's own hatch below, exactly as it always was;
    any other car is lofted from its own body shell (`_body_mesh`)."""
    paint = tuple(int(c) for c in paint)
    dark = (C_PAINT_DARK if paint == C_PAINT
            else tuple(int(round(0.75 * c)) for c in paint))
    if car not in (None, "corsa"):
        return _body_mesh(paint, dark, car)
    polys = []
    inside = np.array([-0.15, 0.0, 0.70])
    rings = [_ring(s) for s in STATIONS]
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        x0, x1 = STATIONS[i][0], STATIONS[i + 1][0]
        for j in range(N_RING):
            k = (j + 1) % N_RING
            quad = np.array([a[j], a[k], b[k], b[j]])
            if j in (3, 4):               # roof
                col = (C_GLASS if _between(x0, x1, X_WINDSCREEN)
                       or _between(x0, x1, X_REAR_GLASS) else paint)
            elif j in (2, 5):             # belt -> roof edge: side glass
                col = C_GLASS if _between(x0, x1, X_SIDE_GLASS) else paint
            elif j in (8, 9):             # floor
                col = C_UNDER
            else:
                col = paint
            polys.append((_orient(quad, inside), col, "body"))
    polys.append((_orient(rings[0], inside), dark, "body"))       # nose
    polys.append((_orient(rings[-1], inside), dark, "body"))      # tail

    # wheel arches: dark discs on the flank, a hair outboard of the sill
    for wx, wy in WHEEL_XY:
        side = 1.0 if wy > 0 else -1.0
        y = side * (CAR_HALF_W + 0.004)
        arch = [(wx + 0.34 * math.cos(t), y, WHEEL_R + 0.34 * math.sin(t))
                for t in np.linspace(0.0, math.pi, 9)]
        arch += [(wx - 0.34, y, 0.16), (wx + 0.34, y, 0.16)]
        polys.append((_orient(arch, inside), C_ARCH, "trim"))

    # wheels: 14-gon cylinders, axis y
    for wx, wy in WHEEL_XY:
        side = 1.0 if wy > 0 else -1.0
        yc = side * WHEEL_Y_DRAW
        centre = np.array([wx, yc, WHEEL_R])
        ts = np.linspace(0.0, 2 * math.pi, 15)[:-1]
        outer = np.array([(wx + WHEEL_R * math.cos(t), yc + side * 0.5 * WHEEL_W,
                           WHEEL_R + WHEEL_R * math.sin(t)) for t in ts])
        inner = outer.copy()
        inner[:, 1] = yc - side * 0.5 * WHEEL_W
        for j in range(len(ts)):
            k = (j + 1) % len(ts)
            quad = np.array([outer[j], outer[k], inner[k], inner[j]])
            polys.append((_orient(quad, centre), C_TYRE, "wheel"))
        polys.append((_orient(outer, centre), C_TYRE, "wheel"))
        polys.append((_orient(inner, centre), C_TYRE, "wheel"))
        rim = outer.copy()
        rim[:, 0] = wx + 0.62 * (rim[:, 0] - wx)
        rim[:, 2] = WHEEL_R + 0.62 * (rim[:, 2] - WHEEL_R)
        rim[:, 1] += side * 0.003
        polys.append((_orient(rim, centre), C_RIM, "wheel"))
    return polys


#: The preview's glass on a lofted shell, read off the shell's own BAND kinds
#: (`bodies._STYLE_SHELL3`): a windscreen or a rear window on the roof quads,
#: side glass between the belt and the roof along the cabin. The Express is
#: a panel van behind its cab (no side glass on the load box: its first two
#: roof bands are the cab), a roadster's cockpit is open (the dark interior),
#: and a Citaro's front face is nearly all windscreen.
_ROOF_GLASS = ("screen", "rglass")
_SIDE_GLASS = ("screen", "roof", "rglass")
_SIDE_GLASS_MAX_BAND = {"van": 4}
#: a wheel's radius when `cars.py` does not carry the car yet (est: the
#: Express's 145R13 / 155R13 van tyre, a Citaro's 275/70R22.5)
_WHEEL_R_EST = {"van": 0.285, "bus": 0.4815}
#: the rally Escort's livery stripes on the preview (task 46): render's
#: C_LIVERY3, the palette's white
C_LIVERY = (226, 226, 220)


def _ring_body(st):
    """`_ring` for a lofted shell: the sill's knee is 0.28 m up the side on
    the Corsa, and on a low bumper face (a van's, a bus's rear) that is above
    the belt, so it is held at most half way from the floor to the belt."""
    x, zb, zbelt, ztop, w, wr = st
    knee = min(zb + 0.28, zb + 0.5 * (zbelt - zb))
    return np.array([
        (x, -0.92 * w, zb), (x, -w, knee), (x, -w, zbelt), (x, -wr, ztop),
        (x, 0.0, ztop + 0.02),
        (x, wr, ztop), (x, w, zbelt), (x, w, knee), (x, 0.92 * w, zb),
        (x, 0.0, zb - 0.02),
    ])


def _body_mesh(paint, dark, car) -> list:
    """The preview of any car but the Corsa: its body shell
    (`bodies.body(car).stations`, the same six numbers per station as
    STATIONS) lofted exactly as the hatch is, glass where its bands say, and
    its own wheels -- `cars.py`'s a / b / t_f / t_r / tyre when it carries
    the car, the style car's axles and an est tyre when it does not yet."""
    b = bodies.body(car)
    spec = bodies._spec(car)
    kinds = bodies._STYLE_SHELL3[b.style][1]
    st = b.stations
    inside = np.array([0.5 * (b.x_front + b.x_rear), 0.0, 0.5 * (b.ground + b.height)])
    rings = [_ring_body(s_) for s_ in st]
    side_max = _SIDE_GLASS_MAX_BAND.get(b.style, len(kinds))
    polys = []
    for i in range(len(rings) - 1):
        a, c = rings[i], rings[i + 1]
        kind = kinds[i] if i < len(kinds) else "roof"
        for j in range(N_RING):
            k = (j + 1) % N_RING
            quad = np.array([a[j], a[k], c[k], c[j]])
            if j in (3, 4) and b.style == "rally" and kind in _LIVERY_KINDS:
                #  task 46: the rally car's twin white stripes, cut INTO the
                #  roof quads (a decal laid over them lost the painter's
                #  sort on the near side): each half of the top split at
                #  |y| 0.26 and 0.10, the middle strip the livery
                polys += _striped_half(a[j], a[k], c[j], c[k], paint, inside)
                continue
            if j in (3, 4):                              # roof
                col = (C_GLASS if kind in _ROOF_GLASS
                       else (C_UNDER if kind == "cockpit" else paint))
            elif j in (2, 5):                            # belt -> roof edge
                col = C_GLASS if (kind in _SIDE_GLASS and i <= side_max) else paint
            elif j in (8, 9):                            # floor
                col = C_UNDER
            else:
                col = paint
            polys.append((_orient(quad, inside), col, "body"))
    polys.append((_orient(rings[0], inside), C_GLASS if b.style == "bus" else dark, "body"))
    polys.append((_orient(rings[-1], inside), dark, "body"))
    a_, L, t_f, t_r = bodies.axles(spec if spec is not None else object(), b.style)
    R = float(getattr(spec, "tyre_R0", 0.0) or _WHEEL_R_EST.get(b.style, WHEEL_R))
    W = float(getattr(spec, "tyre_width", 0.0) or WHEEL_W * R / WHEEL_R)
    wheels = ((a_, 0.5 * t_f), (a_, -0.5 * t_f), (a_ - L, 0.5 * t_r), (a_ - L, -0.5 * t_r))
    arch_r = R + 0.0485                                  # the Corsa's 0.34 m over its 0.2915
    for wx, wy in wheels:
        side = 1.0 if wy > 0 else -1.0
        y = side * (b.half_w_at(wx) + 0.004)
        arch = [(wx + arch_r * math.cos(t), y, R + arch_r * math.sin(t))
                for t in np.linspace(0.0, math.pi, 9)]
        arch += [(wx - arch_r, y, b.ground + 0.01), (wx + arch_r, y, b.ground + 0.01)]
        polys.append((_orient(arch, inside), C_ARCH, "trim"))
    for wx, wy in wheels:
        side = 1.0 if wy > 0 else -1.0
        yc = side * (abs(wy) + 0.0305)                   # the Corsa's 0.745 over its 0.7145
        if b.style == "rally":
            #  task 46: out in the Group 4 arches, flush with them, as the
            #  chase car draws it (render.CarGeom.hub_y); every other car as before
            yc = side * max(abs(wy) + 0.0305, b.half_w_at(wx) - 0.5 * W + 0.010)
        centre = np.array([wx, yc, R])
        ts = np.linspace(0.0, 2 * math.pi, 15)[:-1]
        outer = np.array([(wx + R * math.cos(t), yc + side * 0.5 * W, R + R * math.sin(t))
                          for t in ts])
        inner = outer.copy()
        inner[:, 1] = yc - side * 0.5 * W
        for j in range(len(ts)):
            k = (j + 1) % len(ts)
            quad = np.array([outer[j], outer[k], inner[k], inner[j]])
            polys.append((_orient(quad, centre), C_TYRE, "wheel"))
        polys.append((_orient(outer, centre), C_TYRE, "wheel"))
        polys.append((_orient(inner, centre), C_TYRE, "wheel"))
        rim = outer.copy()
        rim[:, 0] = wx + 0.62 * (rim[:, 0] - wx)
        rim[:, 2] = R + 0.62 * (rim[:, 2] - R)
        rim[:, 1] += side * 0.003
        polys.append((_orient(rim, centre), C_RIM, "wheel"))
    return polys


#: the bands the rally car's stripes run down (the painted top: never glass)
_LIVERY_KINDS = ("bonnet", "roof", "deck")
_LIVERY_Y = (0.10, 0.26)          # m  each stripe's inner and outer edge, |y|


def _striped_half(p0, p1, q0, q1, paint, inside) -> list:
    """One half of a band's top (ring points p0 -> p1 on the band's front
    ring, q0 -> q1 on its rear one; one end at the crown y = 0, the other at
    the roof edge) as three quads: paint, the white stripe between |y|
    _LIVERY_Y, paint. The cuts are placed on each ring's own edge by |y|,
    so a band whose roof narrows keeps the stripe straight (task 46)."""
    def at(a_, b_, y_):
        t = (y_ - abs(a_[1])) / (abs(b_[1]) - abs(a_[1]))
        return a_ + (b_ - a_) * min(max(t, 0.0), 1.0)
    lo, hi = _LIVERY_Y
    fp = [p0] + sorted((at(p0, p1, y_) for y_ in (lo, hi)),
                       key=lambda v: abs(v[1]) if abs(p0[1]) < abs(p1[1]) else -abs(v[1])) + [p1]
    fq = [q0] + sorted((at(q0, q1, y_) for y_ in (lo, hi)),
                       key=lambda v: abs(v[1]) if abs(q0[1]) < abs(q1[1]) else -abs(v[1])) + [q1]
    out = []
    for n in range(3):
        quad = np.array([fp[n], fp[n + 1], fq[n + 1], fq[n]])
        out.append((_orient(quad, inside), C_LIVERY if n == 1 else paint, "body"))
    return out


class PreviewGeo:
    """What the car page draws round, for the car being fitted (task 41):
    the body side the flank panels stand off (`half_w`), the deck a top wing
    stows on (`deck_z`), the car's extents for the ground and the shadow, and
    how far the orbit camera stands back. The Corsa's are the garage's own
    constants and mesh, exactly -- the preview's pixel checks pin them."""

    def __init__(self, car=None):
        self.car = car or "corsa"
        self.corsa = self.car == "corsa"
        if self.corsa:
            self.half_w, self.height = CAR_HALF_W, CAR_H
            self.x_front, self.x_rear = CAR_X_FRONT, CAR_X_REAR
            self.deck_z = deck_z
            self.scale = 1.0
            self.target = (-0.15, 0.0, 0.70)
            self.top_dim_y = 0.5 * 1.7 + 0.15
        else:
            b = bodies.body(self.car)
            self.half_w, self.height = b.half_w, b.height
            self.x_front, self.x_rear = b.x_front, b.x_rear
            self.deck_z = b.deck_z
            #  a 12 m bus has to fit the view the 3.8 m Corsa does
            self.scale = max(1.0, (b.x_front - b.x_rear) / (CAR_X_FRONT - CAR_X_REAR),
                             b.height / CAR_H)
            self.target = (0.5 * (b.x_front + b.x_rear), 0.0, 0.5 * b.height)
            self.top_dim_y = b.half_w + 0.15


class Batch:
    """A polygon soup flattened for one-shot projection: all vertices in one
    (N,3) array, per-polygon start/count, outward normal and centroid. The
    car body is built once; the wings are rebuilt each frame."""

    def __init__(self, polys):
        self.starts = np.zeros(len(polys), dtype=np.int64)
        self.counts = np.zeros(len(polys), dtype=np.int64)
        self.colours = [p[1] for p in polys]
        self.kinds = [p[2] for p in polys]
        vs, n = [], 0
        for i, (v, _, _) in enumerate(polys):
            self.starts[i], self.counts[i] = n, len(v)
            vs.append(v)
            n += len(v)
        self.verts = np.concatenate(vs) if vs else np.zeros((0, 3))
        if polys:
            v0 = self.verts[self.starts]
            v1 = self.verts[self.starts + 1]
            v2 = self.verts[self.starts + 2]
            nrm = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(nrm, axis=1)
            self.normals = nrm / np.where(ln > 1e-12, ln, 1.0)[:, None]
            self.centroids = np.add.reduceat(self.verts, self.starts, axis=0) / self.counts[:, None]
        else:
            self.normals = np.zeros((0, 3))
            self.centroids = np.zeros((0, 3))
        self.no_cull = np.array([k == "flat" for k in self.kinds], dtype=bool)

    @staticmethod
    def join(a: "Batch", b: "Batch") -> "Batch":
        out = Batch.__new__(Batch)
        off = len(a.verts)
        out.starts = np.concatenate([a.starts, b.starts + off])
        out.counts = np.concatenate([a.counts, b.counts])
        out.colours = a.colours + b.colours
        out.kinds = a.kinds + b.kinds
        out.verts = np.concatenate([a.verts, b.verts])
        out.normals = np.concatenate([a.normals, b.normals])
        out.centroids = np.concatenate([a.centroids, b.centroids])
        out.no_cull = np.concatenate([a.no_cull, b.no_cull])
        return out


def _box(x0, x1, y0, y1, z0, z1, col, kind="strut"):
    v = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    centre = v.mean(axis=0)
    faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [(_orient(v[list(f)], centre), col, kind) for f in faces]


_SECTION_CACHE: dict = {}


def _section_loop(lib: "Library | None", name: str, n: int = 11) -> np.ndarray:
    """The section as a (2n-1, 2) loop for lofting (cached per name)."""
    key = (name, n)
    c = _SECTION_CACHE.get(key)
    if c is None:
        from .aero import airfoil as af
        try:
            spec = lib.airfoils[name] if (lib is not None and name in lib.airfoils) else None
            base = spec.coords() if spec is not None else af.naca4_coords("2412")
            c = af.resample(base, n)
        except Exception:
            c = af.naca4_coords("2412", n)
        _SECTION_CACHE[key] = c
    return c


def _loft(rings: list[np.ndarray], col, kind="wing", caps=True) -> list:
    """Quads between consecutive rings. The winding is decided ONCE per band
    (the ring order is consistent along a loft) from the quad at the
    thickest point, which keeps the closed mesh cullable at ~20x less cost
    than orienting every quad."""
    polys = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        centre = 0.5 * (a.mean(axis=0) + b.mean(axis=0))
        j0 = len(a) // 4
        probe = np.array([a[j0], a[j0 + 1], b[j0 + 1], b[j0]])
        flip = np.dot(np.cross(probe[1] - probe[0], probe[2] - probe[0]), probe.mean(axis=0) - centre) < 0.0
        for j in range(len(a) - 1):
            quad = np.array([a[j], a[j + 1], b[j + 1], b[j]])
            polys.append((quad[::-1].copy() if flip else quad, col, kind))
    if caps and rings:
        inside = 0.5 * (rings[0].mean(axis=0) + rings[-1].mean(axis=0))
        polys.append((_orient(rings[0][:-1], inside), col, kind))
        polys.append((_orient(rings[-1][:-1], inside), col, kind))
    return polys


def carried_by_own_plates(spec: "WingSpec | None") -> bool:
    """Is `spec` a WingLab wing its own designed endplates CARRY? AeroBO's
    plate is a fixed arc from the wing's tip to the car, sized at the
    standoff the wing was flown at (`ride_h_flown`)."""
    return spec is not None and spec.engine == "aerobo" and spec.mount == "endplate"


def flank_out(spec: "WingSpec | None", deploy: float, inc_deg: float = 0.0) -> float:
    """A flank panel's standoff from the car's side at this deploy [m]:
    carsim's slide-out (DEV_OUT0 stowed, + DEV_OUT1 deployed). A wing its
    endplates CARRY is deployed at the standoff it was FLOWN at -- a WingLab
    wing's rigid plates are sized to reach the car from there (the owner,
    2026-09-27: "Wing tips not well connected" -- slid out by DEV_OUT1, a
    plate ended in mid-air) -- and stowed it retracts against the side
    (`blend.stowed_standoff` at `inc_deg`), its plates running on into the
    body: plates that carry the wing cannot fold away (the owner, 2026-09-27:
    switched off, they never went away)."""
    from .aero import blend as bl
    d = float(deploy)
    if spec is None or spec.mount != "endplate":
        return DEV_OUT0 + DEV_OUT1 * d
    full = float(spec.ride_h_flown) if carried_by_own_plates(spec) else DEV_OUT0 + DEV_OUT1
    stow = min(bl.stowed_standoff(spec.chord, abs(inc_deg) + abs(spec.twist_deg)), full)
    return stow + (full - stow) * d


def wing_polys(spec: "WingSpec | None", key: str, slot: Slot, deploy: float,
               lib: "Library | None", selected: bool = False, legacy_type: str = "",
               geo: "PreviewGeo | None" = None) -> list:
    """The polygons of one slot's wing at this deploy fraction.

    Flank: a vertical loft of the section (suction side towards the car,
    since its lift is the inward side force), standing off the sill by
    DEV_OUT0 + DEV_OUT1 * deploy (a wing its endplates carry: retracted
    against the side stowed, `flank_out`), optional end plates in the wing's
    colour.
    Top: an inverted loft across the car; stowed it lies on the deck,
    deployed it rises to the slot height and takes its incidence; end
    plates hang towards the road. `geo` is the car being fitted (its body
    side and deck; None: the Corsa).

    WHAT CARRIES IT is the wing's `mount` (the owner, 2026-09-25: "Carried by
    endplate still produces inboard pylons"): 'pylon' draws the two struts
    (flank: to the car side) or swan-neck pylons (top: from the deck behind
    the trailing edge, over onto the pressure surface) at +-`pylon_frac` of
    the semi-span; 'endplate' draws none -- the plates are the structure and
    run from the tips to the car side / the deck along their lean; 'none'
    neither. The plates lean by `plate_cant_deg`, blend out of the wing over
    `plate_blend` of their arc, and carry `plate_chord_ratio` of the tip
    chord or continue the wing's chord law (`plate_chord_follows`) --
    `blend.plate_stations`, which render.wing_mesh3 sweeps too. A wing with
    the defaults (upright, sharp, pylons at 0.56) draws as it always did,
    apart from its pylons' swan neck."""
    from .aero import blend as bl
    from .aero.wing import pylon_rings, plate_chord_scale, lower_surface
    polys = []
    half_w = geo.half_w if geo is not None else CAR_HALF_W
    role = SLOT_ROLE[key]
    if spec is None and not legacy_type:
        return polys
    if spec is None:                                   # the published panel, drawn as before
        chord, span, taper, twist, plate, sec_name = DEV_CHORD, DEV_SPAN, 1.0, 0.0, (0.06 if legacy_type == "plate" else 0.0), ("naca6412" if legacy_type == "plate" else "naca4412")
        mount, pf, cant, blend, shape, scale, follows = "pylon", 0.56, 90.0, 0.0, "arc", plate_chord_scale(role, 0.0, False), False
    else:
        chord, span, taper, twist, plate, sec_name = spec.chord, spec.span, spec.taper, spec.twist_deg, spec.plate_h_flown, spec.airfoil
        mount, pf, cant, blend, shape = spec.mount, spec.pylon_frac, spec.plate_cant_deg, spec.plate_blend, spec.plate_shape
        follows = bool(spec.plate_chord_follows)
        scale = plate_chord_scale(role, spec.plate_chord_ratio, follows)
    #  a CARRYING plate AeroBO designed is its own arc (`endplate_h_m`, reach
    #  checked against AeroBO's flat deck), and carsim BRACKETS it to this
    #  body where it stops short (`blend.carried_stations`); a carsim-designed
    #  carrying plate IS its reach to the body, as it always was drawn
    aerobo_plate = spec is not None and spec.engine == "aerobo"
    own = float(spec.plate_h) if aerobo_plate else PLATE_REACH_MAX
    c_tip = chord * taper
    #  the wing's chord law continued past the tip, per metre of plate arc
    slope = -chord * (1.0 - taper) / max(0.5 * span, 1e-6) if follows else 0.0
    carried = mount == "endplate"

    loop = _section_loop(lib, sec_name)
    on = bool(spec is not None or legacy_type)
    col = C_PANEL_SEL if selected else (C_PANEL_ON if on else C_PANEL_OFF)
    n_st = 6
    if role == "flank":
        s = 1.0 if key == "left" else -1.0
        out = flank_out(spec, deploy, slot.inc_deg)
        yc = s * (half_w + out)
        rings = []
        for i in range(n_st):
            eta = -1.0 + 2.0 * i / (n_st - 1)          # -1 bottom .. +1 top
            c = chord * (1.0 - (1.0 - taper) * abs(eta))
            z = slot.h + eta * 0.5 * span
            th = s * math.radians(slot.inc_deg + twist * abs(eta))
            ct, st_ = math.cos(th), math.sin(th)
            pts = []
            for xa, ya in loop:
                dx = (0.5 - xa) * c
                dy = -s * ya * c
                pts.append((slot.x + dx * ct - dy * st_, yc + dx * st_ + dy * ct, z))
            rings.append(np.array(pts))
        polys += _loft(rings, col, "wingsel" if selected else "wing")
        if mount == "pylon":                       # two struts to the car side
            #  task 46: each lands on the body at its height (`side_land`)
            for dz in (-pf * 0.5 * span, pf * 0.5 * span):
                polys += _flank_strut(geo, slot.x, yc, slot.h + dz, s, C_STRUT)
        if plate > 0.0 or carried:
            #  a fence crosses the tip (half each side, as carsim has always
            #  drawn it) when sharp; blended it grows out of the tip towards
            #  the car; carrying the panel it runs to the car side -- AeroBO's
            #  plate, then a bracket where it stops short
            br = None
            if carried:
                st, br, reached = bl.carried_stations(own, lambda _dy: out, cant, blend, shape,
                                                      c_tip, scale, slope, 0.0, PLATE_REACH_MAX)
            else:
                #  WingLab's fence (a WingLab wing on the pylons) is flown
                #  whole towards the car -- its clearance margin charges all
                #  of it there -- so it is drawn so, with the top wing's stub
                #  past the tip; carsim's own fence crosses the tip
                h_d = plate
                back = 0.0 if blend > 0.0 else (0.03 if aerobo_plate else 0.5 * h_d)
                h_d = h_d if (blend > 0.0 or aerobo_plate) else 0.5 * h_d
                cap = h_d
                if h_d > out:                      # a fence stops AT the car's side
                    h_d = bl.reach_arc(lambda _dy: out, cant, blend, shape, cap)
                st = bl.plate_stations(h_d, cant, blend, shape, c_tip, scale, slope, back)
                reached = h_d < cap - 1e-9
            #  set down on the side only where it got there (a lean too
            #  shallow to reach is drawn falling short, not bent onto it)
            #  (task 46: on the side the body has at that corner's station --
            #  narrower at the nose and the tail than its widest line; a foot
            #  above the belt is stayed down onto it below)
            wall = ((1, lambda p_: s * side_land(geo, p_[0], min(p_[2], _section_at(geo, p_[0])[2]))[0])
                    if reached else None)
            #  a carrying plate whose foot is past the drawn side -- above
            #  its belt (a bonnet's edge at the front wheels) or below its
            #  bottom -- is stayed onto it (`blend.side_stay`)
            top_at, z_bot = _deck_top(geo, slot.x), _body_bottom(geo, slot.x)
            #  task 46: a tip device (a fence, not a carrying plate) is fixed
            #  to the tip section, so it turns with the section's incidence
            #  and twist -- along x it left the turned section half off it
            stand, cdir = (0.0, -s, 0.0), None
            if not carried:
                th_t = s * math.radians(slot.inc_deg + twist)
                cdir = (math.cos(th_t), math.sin(th_t), 0.0)
                stand = (s * math.sin(th_t), -s * math.cos(th_t), 0.0)
            for sgn in (-1.0, 1.0):
                root = (slot.x, yc, slot.h + sgn * 0.5 * span)
                rg = bl.plate_rings(root, (0.0, 0.0, sgn), stand, st, 0.012,
                                    None if br is not None else wall, chord_dir=cdir)
                polys += _loft(rg, col, "plate")          # the wing's own colour
                if br is not None:
                    polys += _loft(bl.plate_rings(root, (0.0, 0.0, sgn), (0.0, -s, 0.0), br,
                                                  0.012, wall), C_STRUT, "bracket")
                if carried:
                    foot = bl.plate_foot(root, (0.0, 0.0, sgn), (0.0, -s, 0.0),
                                         br if br is not None else st)
                    #  (task 46: where the wall set the foot down, at this
                    #  station's side, not the widest line)
                    fy = (s * side_land(geo, slot.x, min(foot[2], _section_at(geo, slot.x)[2]))[0]
                          if wall is not None else foot[1])
                    stay = bl.side_stay(foot[2], top_at(fy), z_bot)
                    if stay is not None:
                        cw = 0.5 * max(bl.BRACKET_CHORD_FRAC * float(st["c"][-1]),
                                       bl.BRACKET_CHORD_MIN)
                        polys += _box(slot.x - cw, slot.x + cw, fy - 0.006, fy + 0.006,
                                      stay[0], stay[1], C_STRUT, "bracket")
    else:
        deck = (geo.deck_z if geo is not None else deck_z)(slot.x)
        z_stow = deck + TOP_STOW_GAP
        zc = z_stow + (slot.h - z_stow) * deploy
        ang = math.radians(slot.inc_deg) * deploy
        rings = []
        n_st = 7
        for i in range(n_st):
            eta = -1.0 + 2.0 * i / (n_st - 1)
            y = eta * 0.5 * span
            c = chord * (1.0 - (1.0 - taper) * abs(eta))
            a = ang + math.radians(twist * abs(eta)) * deploy
            ca, sa = math.cos(a), math.sin(a)
            pts = []
            for xa, ya in loop:
                dx = (0.5 - xa) * c
                dz = -ya * c                          # inverted: suction side down
                pts.append((slot.x + dx * ca + dz * sa, y, zc - dx * sa + dz * ca))
            rings.append(np.array(pts))
        polys += _loft(rings, col, "wingsel" if selected else "wing")
        z_tip = zc - 0.5 * span * math.sin(ang) * 0.0
        if plate > 0.0 or carried:
            #  hanging towards the road; a fence folds up with the stowed
            #  wing, a carrying plate always reaches the body
            fold = plate * deploy + 0.02 * (1 - deploy)
            #  carrying, it lands on the body where its lean takes it: on a
            #  roof, or down the shoulder a wide wing's tips stand over
            top_at = _deck_top(geo, slot.x)

            def drop(dy):
                return z_tip - top_at(0.5 * span + 0.008 + dy)
            back = 0.0 if blend > 0.0 else 0.03
            br = None
            if carried:
                #  AeroBO's plate, bracketed to the body where it stops short
                st, br, reached = bl.carried_stations(own, drop, cant, blend, shape, c_tip,
                                                      scale, slope, back, PLATE_REACH_MAX)
            else:
                h_d = fold
                if h_d > drop(0.0):                # ...and a fence stops ON the body
                    h_d = bl.reach_arc(drop, cant, blend, shape, fold)
                st = bl.plate_stations(h_d, cant, blend, shape, c_tip, scale, slope, back)
                reached = h_d < fold - 1e-9
            wall = (2, lambda p_: _deck_top(geo, p_[0])(p_[1])) if reached else None
            #  task 46: a tip device turns with the tip section (the flank's rule)
            stand, cdir = (0.0, 0.0, -1.0), None
            if not carried:
                a_t = ang + math.radians(twist) * deploy
                cdir = (math.cos(a_t), 0.0, -math.sin(a_t))
                stand = (-math.sin(a_t), 0.0, -math.cos(a_t))
            for sgn in (-1.0, 1.0):
                root = (slot.x, sgn * (0.5 * span + 0.008), z_tip)
                rg = bl.plate_rings(root, (0.0, sgn, 0.0), stand, st, 0.012,
                                    None if br is not None else wall, chord_dir=cdir)
                polys += _loft(rg, col, "plate")
                if br is not None:
                    polys += _loft(bl.plate_rings(root, (0.0, sgn, 0.0), (0.0, 0.0, -1.0), br,
                                                  0.012, wall), C_STRUT, "bracket")
                if carried and wall is not None:
                    #  task 46: a wing wider than the body sets its plates
                    #  down BESIDE it, at the belt (`_deck_top` past the
                    #  side): a bracket runs in from the foot to the side
                    foot = bl.plate_foot(root, (0.0, sgn, 0.0), (0.0, 0.0, -1.0),
                                         br if br is not None else st)
                    _zb, _kn, zbelt, _zt, w_x, _wr = _section_at(geo, slot.x)
                    if abs(foot[1]) > w_x + 0.01:
                        cw = 0.5 * max(bl.BRACKET_CHORD_FRAC * float(st["c"][-1]),
                                       bl.BRACKET_CHORD_MIN)
                        polys += _box(slot.x - cw, slot.x + cw, min(sgn * w_x, foot[1]),
                                      max(sgn * w_x, foot[1]), zbelt - 0.012, zbelt,
                                      C_STRUT, "sidebracket")
        if mount == "pylon":
            #  swan necks: the section at the pylon's station, its PRESSURE
            #  surface (the world-up side of the inverted wing) for the neck
            c_p = chord * (1.0 - (1.0 - taper) * pf)
            a = ang + math.radians(twist * pf) * deploy
            ca, sa = math.cos(a), math.sin(a)
            up = lower_surface(loop)
            dx = (0.5 - up[:, 0]) * c_p
            dz = -up[:, 1] * c_p
            px, pz = slot.x + dx * ca + dz * sa, zc - dx * sa + dz * ca
            for sgn in (-1.0, 1.0):
                polys += _loft(pylon_rings(px, pz, sgn * pf * 0.5 * span,
                                           geo.deck_z if geo is not None else deck_z),
                               C_STRUT, "strut")
    return polys


#: a carrying plate's arc is capped here: a lean so shallow that reaching the
#: body would take more is drawn falling short -- AeroBO's own reach floor
#: (`nice_app._car_plate_cant_reach_floor`) refuses that layout anyway
PLATE_REACH_MAX = 2.0


def _deck_top(geo, x: float):
    """The drawn body's TOP across station x, as y -> z: where a carrying
    top-wing plate lands. The section `_ring` lofts -- the crown, the roof
    edge at `wr`, down the glass to the belt at the side `w` -- interpolated
    between stations; past the side, the belt (the plate stands beside the
    car at its waist). A callable, so a plate's reach solve reads one
    station's profile many times for the price of one."""
    rows = STATIONS if (geo is None or geo.corsa) else bodies.body(geo.car).stations
    st = np.array(rows, float)[::-1]                      # x ascending
    _x, _zb, zbelt, ztop, w, wr = (float(np.interp(x, st[:, 0], st[:, k])) for k in range(6))
    ys, zs = (0.0, wr, w), (ztop + 0.02, ztop, zbelt)
    return lambda y: float(np.interp(abs(y), ys, zs))


def _body_bottom(geo, x: float) -> float:
    """The drawn body's BOTTOM at station x (its stations' `zb`): below it a
    flank plate's foot has no side to stand on."""
    rows = STATIONS if (geo is None or geo.corsa) else bodies.body(geo.car).stations
    st = np.array([r[:6] for r in rows], float)[::-1]      # x ascending
    return float(np.interp(x, st[:, 0], st[:, 1]))


def _section_at(geo, x: float) -> tuple:
    """The drawn body's cross-section at station x: (zb, knee, zbelt, ztop,
    w, wr) -- `_ring` (the Corsa) / `_ring_body` (a lofted shell) read
    between stations: the floor corner at 0.92 w, the side upright at w
    from the sill's knee to the belt, the glass in to the roof edge wr."""
    corsa = geo is None or geo.corsa
    rows = STATIONS if corsa else bodies.body(geo.car).stations
    st = np.array([r[:6] for r in rows], float)[::-1]      # x ascending
    _x, zb, zbelt, ztop, w, wr = (float(np.interp(x, st[:, 0], st[:, k])) for k in range(6))
    knee = zb + 0.28 if corsa else min(zb + 0.28, zb + 0.5 * (zbelt - zb))
    return zb, min(knee, zbelt - 1e-6), zbelt, max(ztop, zbelt + 1e-6), w, wr


def side_land(geo, x: float, z: float) -> tuple[float, float]:
    """Where a flank strut meets the drawn body at station x, height z, as
    (|y|, z) -- task 46 (the owner, 2026-09-28: "when pushing the wing
    forward it sometimes is floating in the air. Make it so that ... both
    pylons ... are both connected to a physical part of the car"). Level
    with a side, the side at that height: the upright flank, the sill's
    tuck below the knee, the glass (tumblehome) above the belt. Above the
    body's top there (over a bonnet, past a tail) the strut braces down to
    its top edge; below its floor, up to the floor corner. A strut drawn to
    the widest half-width at every height ended in mid-air beside the glass
    and over the bonnet."""
    zb, knee, zbelt, ztop, w, wr = _section_at(geo, x)
    if z > ztop:
        #  the nearer of the two top corners: a bonnet's wing edge (the belt
        #  at w), else the roof edge
        return (w, zbelt) if (z - zbelt) ** 2 < (z - ztop) ** 2 + (w - wr) ** 2 else (wr, ztop)
    if z < zb:
        return 0.92 * w, zb
    return float(np.interp(z, (zb, knee, zbelt, ztop), (0.92 * w, w, w, wr))), float(z)


def _beam(p0, p1, hw_x: float, ht: float, col, kind="strut") -> list:
    """A straight strut of `2 hw_x` along x and `2 ht` across, from p0 to
    p1 in the y-z plane (a flank strut's brace, a plate's side bracket)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    ex = np.array([1.0, 0.0, 0.0])
    n = np.cross(ex, d)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    rings = []
    for p in (p0, p1):
        rings.append(np.array([p + hw_x * ex + ht * n, p - hw_x * ex + ht * n,
                               p - hw_x * ex - ht * n, p + hw_x * ex - ht * n,
                               p + hw_x * ex + ht * n]))
    return _loft(rings, col, kind)


def _flank_strut(geo, x: float, y_wing: float, z: float, s: float, col) -> list:
    """One flank pylon strut from the wing (at |y_wing|, height z) to the
    body (`side_land`): the level box it always was where the body has a
    side at that height, a brace to the body's top edge (or floor) where it
    has none."""
    y_b, z_b = side_land(geo, x, z)
    if abs(z_b - z) < 1e-9:
        return _box(x - 0.015, x + 0.015, min(s * y_b, y_wing), max(s * y_b, y_wing),
                    z - 0.012, z + 0.012, col)
    return _beam((x, s * y_b, z_b), (x, y_wing, z), 0.015, 0.012, col)


def _hull2(p: np.ndarray) -> np.ndarray:
    """The convex hull of 2-D points, counter-clockwise (monotone chain)."""
    p = np.unique(np.round(np.asarray(p, float), 12), axis=0)
    if len(p) < 3:
        return p

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0.0:
                h.pop()
            h.append(q)
        return h
    lo, hi = half(p), half(p[::-1])
    return np.array(lo[:-1] + hi[:-1])


def tip_cover(polys, role: str) -> float:
    """The share of a wing's +tip section (its outermost ring along the span)
    that lies on its tip device: the plate's drawn vertices near the tip,
    seen along the span, as a convex outline (task 46's check: the plate
    along x left a turned section half off it)."""
    ax = 2 if role == "flank" else 1
    wv = np.vstack([np.asarray(v, float) for v, _c, k in polys if k in ("wing", "wingsel")])
    tip = wv[np.abs(wv[:, ax] - wv[:, ax].max()) < 1e-9]
    pl = [np.asarray(v, float) for v, _c, k in polys if k == "plate"]
    if not pl:
        return 0.0
    pv = np.vstack(pl)
    pv = pv[np.abs(pv[:, ax] - wv[:, ax].max()) < 0.03]
    keep = [k for k in range(3) if k != ax]
    hull = _hull2(pv[:, keep])
    if len(hull) < 3:
        return 0.0
    inside = 0
    e = np.roll(hull, -1, axis=0) - hull
    el = np.maximum(np.hypot(e[:, 0], e[:, 1]), 1e-12)
    for q in tip[:, keep]:
        #  signed distance inside each edge; 1 mm of slack: a plate that
        #  follows the chord ends exactly AT the tip's leading and trailing edges
        c = (e[:, 0] * (q[1] - hull[:, 1]) - e[:, 1] * (q[0] - hull[:, 0])) / el
        inside += int(np.all(c >= -1e-3))
    return inside / max(len(tip), 1)


def tip_turn_error(polys, role: str) -> float:
    """Degrees between a wing's +tip chord line (its tip section's two
    farthest points) and its plate's first ring there (the plate vertices
    level with the tip along the span): 0 when a blended tip device grows
    out of the tip section as it is turned."""
    ax = 2 if role == "flank" else 1
    wv = np.vstack([np.asarray(v, float) for v, _c, k in polys if k in ("wing", "wingsel")])
    tip = wv[np.abs(wv[:, ax] - wv[:, ax].max()) < 1e-9]
    pv = np.vstack([np.asarray(v, float) for v, _c, k in polys if k == "plate"])
    ring = pv[np.abs(pv[:, ax] - wv[:, ax].max()) < 0.009]
    if len(ring) < 2 or len(tip) < 2:
        return 90.0

    def along(q):
        d = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        u = q[j] - q[i]
        u[ax] = 0.0
        return u / max(float(np.linalg.norm(u)), 1e-12)
    r_ = ring - ring.mean(axis=0)          # the ring's long axis: its corners
    r_[:, ax] = 0.0                         # sit either side of it by the thickness
    u_r = np.linalg.svd(r_, full_matrices=False)[2][0]
    c = abs(float(np.dot(along(tip), u_r)))
    return math.degrees(math.acos(min(1.0, c)))


def _mesh_section(mesh: "Batch", x: float) -> list:
    """The car mesh sliced by the plane x = const: (|y|, z) segments."""
    segs = []
    for i in range(len(mesh.starts)):
        if mesh.kinds[i] != "body":
            continue
        v = mesh.verts[mesh.starts[i]:mesh.starts[i] + mesh.counts[i]]
        if v[:, 0].min() > x or v[:, 0].max() < x or np.ptp(v[:, 0]) < 1e-9:
            continue
        w = np.roll(v, -1, axis=0)
        dx = w[:, 0] - v[:, 0]
        ok = ((v[:, 0] - x) * (w[:, 0] - x) <= 0.0) & (np.abs(dx) > 1e-12)
        if ok.sum() >= 2:
            t = (x - v[ok, 0]) / dx[ok]
            q = np.column_stack([np.abs(v[ok, 1] + t * (w[ok, 1] - v[ok, 1])),
                                 v[ok, 2] + t * (w[ok, 2] - v[ok, 2])])
            segs.append((q[0], q[1]))
    return segs


def _seg_dist2(segs, y: float, z: float) -> float:
    q, best = np.array([abs(y), z]), 9.0
    for a, b in segs:
        d = b - a
        t = float(np.clip(np.dot(q - a, d) / max(float(np.dot(d, d)), 1e-18), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(q - (a + t * d))))
    return best


def _wing_drawing_checks(lib=None, view=None, screenshot_dir: str = "") -> list:
    """The mount and the plates AS DRAWN (the owner, 2026-09-25: "Carried by
    endplate still produces inboard pylons"), measured off `wing_polys`'
    polygons rather than read off the flags that built them: (tag, ok, msg)
    rows for `self_check`, runnable on their own (`view` None: no pictures).
    The test wings are built by hand from the WingSpec contract rows, the way
    the AeroBO bridge fills them: AeroBO's inboard station 0.35, a 70 deg
    lean, a 0.4 spiral blend, a 1.8 chord ratio, the chord law followed on
    a 0.6 taper."""
    from .aero import blend as bl
    rows = []

    def rep(tag, ok, msg=""):
        rows.append((tag, bool(ok), msg))

    def spec_of(role, **kw):
        d = dict(role=role, span=(1.40 if role == "top" else 0.78),
                 chord=(0.30 if role == "top" else 0.45), plate_h=0.12, airfoil="naca4412")
        d.update(kw)
        return WingSpec(**d).clamp()

    top_slot = Slot("t", -0.90, 1.55, 6.0, "active")
    fl_slot = Slot("f", 0.97, 0.90, 0.0)

    def parts(spec, key, deploy=1.0):
        out = {"strut": [], "plate": [], "wing": [], "wingsel": [], "bracket": []}
        for v, _c, kd in wing_polys(spec, key, top_slot if key == "top" else fl_slot, deploy, lib):
            out.setdefault(kd, []).append(np.asarray(v, float))
        return out

    def plate_ends(polys, sgn, zc=1.55, turned=False):
        """(root ring, far ring) of a top wing's plate on the `sgn` side, as
        (centre, chord): the 4 corners nearest its junction (0.708 m out, at
        the wing's height `zc`) and the 4 farthest, across the span."""
        v = np.vstack([p for p in polys if np.mean(p[:, 1]) * sgn > 0])
        v = np.unique(np.round(v, 9), axis=0)
        o = np.argsort(np.hypot(v[:, 1] - sgn * 0.708, v[:, 2] - zc))
        if turned:
            #  task 46: a tip device turns with the tip section by the wing's
            #  6 deg, so its rings are measured in its own frame: turned
            #  back about the junction first, then read as before
            a_ = math.radians(top_slot.inc_deg)
            dx_, dz_ = v[:, 0] - top_slot.x, v[:, 2] - zc
            v = np.column_stack([top_slot.x + dx_ * math.cos(a_) - dz_ * math.sin(a_), v[:, 1],
                                 zc + dx_ * math.sin(a_) + dz_ * math.cos(a_)])
            o = np.argsort(np.hypot(v[:, 1] - sgn * 0.708, v[:, 2] - zc))
        near, far = v[o[:4]], v[o[-4:]]
        return ((near.mean(axis=0), float(np.ptp(near[:, 0]))),
                (far.mean(axis=0), float(np.ptp(far[:, 0]))))

    # -- 1. what carries it: struts / pylons ONLY under pylons --------------
    cnt, st_y, st_z = {}, [], []
    for m in ("pylon", "endplate", "none"):
        pt = parts(spec_of("top", mount=m, pylon_frac=0.35), "top")
        pf_ = parts(spec_of("flank", mount=m, pylon_frac=0.35), "left")
        cnt[m] = (len(pt["strut"]), len(pf_["strut"]))
        if m == "pylon":
            yv = np.vstack(pt["strut"])[:, 1]
            st_y = [float(yv[yv > 0].mean()), float(-yv[yv < 0].mean())]
            #  each strut's height at the PANEL's end (task 46: one with no
            #  side level with it braces down to the body's top edge)
            st_z = []
            for i_ in range(0, len(pf_["strut"]), 6):
                sv = np.vstack(pf_["strut"][i_:i_ + 6])
                sv = sv[np.abs(sv[:, 1]) >= np.abs(sv[:, 1]).max() - 0.015]   # a brace's end ring leans
                st_z.append(float(sv[:, 2].mean()))
            st_z.sort()
    y_want, z_want = 0.35 * 0.70, 0.35 * 0.39
    rep("wing drawing: pylons / struts ONLY under a pylon mount, at +-pylon_frac of the "
        "semi-span; an endplate mount and 'none' draw none",
        0 < cnt["pylon"][0] <= 60 and cnt["pylon"][0] % 2 == 0 and cnt["pylon"][1] == 12
        and cnt["endplate"] == (0, 0) and cnt["none"] == (0, 0)
        and all(abs(y_ - y_want) < 0.002 for y_ in st_y)
        and len(st_z) == 2 and abs(st_z[0] - (0.90 - z_want)) < 1e-9
        and abs(st_z[1] - (0.90 + z_want)) < 1e-9,
        f"top/flank strut polys: pylon {cnt['pylon']}, endplate {cnt['endplate']}, none "
        f"{cnt['none']}; top pylons at y +-{st_y[0]:.4f}/{st_y[1]:.4f} m (0.35 x 0.70 = "
        f"{y_want:.4f}), flank struts at z {st_z[0]:.4f}/{st_z[1]:.4f}")

    # -- 2. the swan neck: deck, behind the TE, over the top, onto it -----
    pt = parts(spec_of("top", mount="pylon", pylon_frac=0.35), "top")
    wv = np.vstack(pt["wing"])
    x_te = float(wv[:, 0].min())
    ok_sw, why = True, []
    for sgn in (1.0, -1.0):
        pv = np.vstack([p for p in pt["strut"] if np.mean(p[:, 1]) * sgn > 0])
        uv = np.unique(np.round(pv, 9), axis=0)
        foot = uv[np.argsort(uv[:, 2])[:4]]                  # its 4 lowest corners
        gap = max(abs(float(z_) - deck_z(float(x_))) for x_, _y, z_ in foot)
        near = wv[np.abs(wv[:, 1] - sgn * y_want) < 0.12]
        fwd = pv[pv[:, 0] > x_te + 0.01]
        ok_sw &= (float(foot[:, 0].max()) < x_te and gap < 1e-9
                  and float(pv[:, 2].max()) > float(near[:, 2].max())
                  and float(fwd[:, 2].min()) > top_slot.h - 0.01
                  and float(pv[:, 0].max()) > x_te + 0.5 * 0.3)
        why.append(f"foot x {float(foot[:, 0].max()):+.3f} (TE {x_te:+.3f}) on the deck to "
                   f"{gap:.1e} m, neck top {float(pv[:, 2].max()):.3f} over the wing's "
                   f"{float(near[:, 2].max()):.3f}, lowest point ahead of the TE "
                   f"{float(fwd[:, 2].min()):.3f} (mid-plane {top_slot.h:.2f})")
    rep("wing drawing: a top wing's pylon is a swan neck -- from the deck behind the "
        "trailing edge, over the top, down onto the pressure (upper) surface",
        ok_sw, "; ".join(why[:1]))

    # -- 3. an endplate mount's plates CARRY: they reach the body ----------
    #  measured against the DRAWN car (a vertical ray on the body mesh's
    #  upward faces), not the profile the plates were landed with
    car = Batch(build_car_mesh())
    up_ = [i for i in range(len(car.starts)) if car.kinds[i] == "body" and car.normals[i][2] > 0.2]

    def body_top(x, y):
        best = -math.inf
        for i in up_:
            v = car.verts[car.starts[i]:car.starts[i] + car.counts[i]]
            w_ = np.roll(v, -1, axis=0)
            cr = (v[:, 1] > y) != (w_[:, 1] > y)
            if not cr.any():
                continue
            xs_ = v[cr, 0] + (y - v[cr, 1]) * (w_[cr, 0] - v[cr, 0]) / (w_[cr, 1] - v[cr, 1])
            if int((xs_ > x).sum()) % 2 == 0:
                continue
            n_, c_ = car.normals[i], car.centroids[i]
            best = max(best, float(c_[2] - (n_[0] * (x - c_[0]) + n_[1] * (y - c_[1])) / n_[2]))
        return best

    st_ = np.array(STATIONS, float)[::-1]

    def waist(x):                  # the body's widest line: (half-width, height)
        return (float(np.interp(x, st_[:, 0], st_[:, 4])), float(np.interp(x, st_[:, 0], st_[:, 2])))

    ok_r, msg_r = True, []
    for kw, span_ in ((dict(), 1.40), (dict(plate_cant_deg=70.0, plate_blend=0.4, plate_shape="spiral",
                                            plate_chord_ratio=1.8), 1.40),
                      (dict(plate_cant_deg=70.0, plate_blend=0.4, plate_shape="spiral"), 1.10)):
        for dep in (1.0, 0.0):
            pt = parts(spec_of("top", mount="endplate", span=span_, **kw), "top", dep)
            gaps, beside = [], []
            for sgn in (1.0, -1.0):
                pv = np.vstack([p for p in pt["plate"] if np.mean(p[:, 1]) * sgn > 0])
                uv = np.unique(np.round(pv, 9), axis=0)
                zc_ = 1.55 if dep else deck_z(-0.9) + TOP_STOW_GAP
                d_ = np.hypot(uv[:, 1] - sgn * (0.5 * span_ + 0.008), uv[:, 2] - zc_)
                for x_, y_, z_ in uv[np.argsort(d_)[-4:]]:             # its foot
                    zt_ = body_top(x_, y_)
                    if zt_ > -math.inf:
                        gaps.append(z_ - zt_)                          # on the body
                    else:                                              # past its side
                        w_, zw_ = waist(x_)
                        beside.append((abs(y_) - w_, z_ - zw_))
            fl = parts(spec_of("flank", mount="endplate", **kw), "left", dep)
            fy = np.vstack(fl["plate"])[:, 1]
            #  (task 46: a flank plate's foot is on the side at its own
            #  station and height -- the nose narrows, and low down the
            #  sill tucks in to 0.92 of the widest line)
            ok_r &= (all(abs(g_) < 0.02 for g_ in gaps)
                     and all(dy_ > 0.0 and abs(dz_) < 1e-9 for dy_, dz_ in beside)
                     and len(gaps) + len(beside) == 8
                     and 0.88 * CAR_HALF_W <= float(fy.min()) <= CAR_HALF_W + 1e-9)
            what = ("upright" if not kw else "lean 70 + blend") + f" b {span_:.2f} dep {dep:.0f}"
            msg_r.append(what + (f": on the body to {max(abs(g_) for g_ in gaps):.3f} m" if gaps else "")
                         + (f": beside it at its waist, {max(d_[0] for d_ in beside):.2f} m out"
                            if beside else ""))
    msg_r.append(f"flank plates to the car's side (y {CAR_HALF_W:.3f}, its sill's tuck lower down) in every case")
    fence = np.vstack(parts(spec_of("top", mount="pylon"), "top")["plate"])
    f_low = top_slot.h - float(fence[:, 2].min())
    #  task 46: the fence turns with the tip section, so its leading corner
    #  hangs lowest: its height's cos plus its half chord's sin at the 6 deg
    from .aero.wing import plate_chord_scale
    a6 = math.radians(top_slot.inc_deg)
    f_want = 0.12 * math.cos(a6) + 0.5 * 0.30 * plate_chord_scale("top", 0.0, False) * math.sin(a6)
    rep("wing drawing: an endplate mount's plates run to the body (top: onto the roof or "
        "down its shoulder, measured on the drawn car; a lean that throws them past its "
        "side stands them beside it at its waist) and the car's side (flank), deployed "
        "and stowed; a pylon wing's fences hang their own height (turned with the tip)",
        ok_r and abs(f_low - f_want) < 1e-9,
        "; ".join(msg_r) + f"; pylon fence {f_low:.3f} m (0.12 m turned 6 deg: {f_want:.3f})")

    #  a FENCE (pylons carry the wing) longer than its gap to the roof stops
    #  on the roof instead of passing through it; stowed it folds away
    through = []
    for dep in (1.0, 0.0):
        pt = parts(spec_of("top", mount="pylon", span=1.0, plate_h=0.30), "top", dep)
        pv = np.unique(np.round(np.vstack(pt["plate"]), 9), axis=0)
        through.append(min(z_ - body_top(x_, y_) for x_, y_, z_ in pv
                           if body_top(x_, y_) > -math.inf))
    rep("wing drawing: a fence longer than its wing's gap to the roof stops ON the roof "
        "(it used to be drawn 0.18 m through it); stowed it folds clear",
        abs(through[0]) <= 0.02 and through[1] >= -0.02,
        f"0.30 m plates under a wing {top_slot.h - deck_z(-0.9):.2f} m over the roof: lowest "
        f"corner {through[0]:+.3f} m off the drawn roof deployed, {through[1]:+.3f} stowed")

    # -- 4. the LEAN throws the far end outboard by h cos(cant) -------------
    def far_dy(**kw):
        pp = parts(spec_of("top", mount="pylon", **kw), "top")["plate"]
        _r, (f_, _fc) = plate_ends(pp, 1.0, turned=True)
        _r2, (f2, _fc2) = plate_ends(pp, -1.0, turned=True)
        return float(f_[1]) - (0.70 + 0.008), -float(f2[1]) - (0.70 + 0.008)
    d90, d70 = far_dy(), far_dy(plate_cant_deg=70.0)
    d70b = far_dy(plate_cant_deg=70.0, plate_blend=0.4, plate_shape="spiral")
    want70 = 0.12 * math.cos(math.radians(70.0))
    wantb = bl.projection(0.12, 70.0, 0.4, "spiral")
    rep("wing drawing: a plate leaning at 70 deg reaches OUTBOARD by h cos(cant) on both "
        "sides; a blend throws it further (blend.projection), upright reaches nothing",
        abs(d90[0]) < 1e-9 and abs(d90[1]) < 1e-9
        and all(abs(d_ - want70) < 1e-6 for d_ in d70)
        and all(abs(d_ - wantb) < 1e-6 for d_ in d70b) and wantb > want70,
        f"upright {d90[0]:+.4f}, lean 70 {d70[0]:+.4f}/{d70[1]:+.4f} (0.12 cos 70 = "
        f"{want70:.4f}), + blend 0.4 {d70b[0]:+.4f} ({wantb:.4f})")

    # -- 5. the CHORD: the ratio, the ramp, the wing's chord law followed --
    c_tip = 0.30 * 0.6
    pp = parts(spec_of("top", mount="pylon", taper=0.6, plate_cant_deg=70.0,
                       plate_chord_follows=True), "top")["plate"]
    (_r, c_root), (_f, c_far) = plate_ends(pp, 1.0, turned=True)
    c_want = c_tip - 0.30 * 0.4 * 0.12 / 0.70
    pp0 = parts(spec_of("top", mount="pylon", taper=0.6, plate_cant_deg=70.0), "top")["plate"]
    (_r0, c_root0), (_f0, c_far0) = plate_ends(pp0, 1.0, turned=True)
    rep("wing drawing: a pylon wing's tip device that FOLLOWS the chord continues the "
        "wing's taper down the plate (from the tip chord); one that does not holds 1.3 x",
        abs(c_root - c_tip) < 1e-9 and abs(c_far - c_want) < 1e-9 and c_far < c_root
        and abs(c_root0 - 1.3 * c_tip) < 1e-9 and abs(c_far0 - c_root0) < 1e-9,
        f"follows: {c_root:.4f} -> {c_far:.4f} m (law {c_want:.4f}); held: {c_root0:.4f} -> "
        f"{c_far0:.4f}")
    pr = parts(spec_of("top", mount="endplate", taper=0.6, plate_chord_ratio=1.8), "top")["plate"]
    (_a, cr_sharp), _b = plate_ends(pr, 1.0)
    prb = parts(spec_of("top", mount="endplate", taper=0.6, plate_chord_ratio=1.8,
                        plate_blend=0.4, plate_shape="spiral", plate_cant_deg=70.0), "top")["plate"]
    (_a2, cr_bl), (_b2, cf_bl) = plate_ends(prb, 1.0)
    rep("wing drawing: the designed plate's chord is its ratio x the tip chord; a blend "
        "ramps it in from the tip chord over the turn (WingLab's lattice law)",
        abs(cr_sharp - 1.8 * c_tip) < 1e-9 and abs(cr_bl - c_tip) < 1e-9
        and abs(cf_bl - 1.8 * c_tip) < 1e-9,
        f"sharp {cr_sharp:.4f} (1.8 x {c_tip:.3f} = {1.8 * c_tip:.4f}); blended root "
        f"{cr_bl:.4f} -> far {cf_bl:.4f}")

    #  the pylon mount's four tip devices (AeroBO's none | vertical | canted |
    #  blended), each following the chord of a 0.6-taper wing
    tips = {}
    for tag, kw in (("none", dict(plate_h=0.0)), ("vertical", {}),
                    ("canted", dict(plate_cant_deg=70.0)),
                    ("blended", dict(plate_blend=0.5, plate_shape="spiral"))):
        pp = parts(spec_of("top", mount="pylon", pylon_frac=0.35, taper=0.6,
                           plate_chord_follows=True, **kw), "top")["plate"]
        tips[tag] = (len(pp), plate_ends(pp, 1.0, turned=True) if pp else None)
    (_rv, _cv), (fv, cfv) = tips["vertical"][1]
    (_rc, _cc), (fc, cfc) = tips["canted"][1]
    (_rb, crb), (fb, cfb) = tips["blended"][1]
    rep("wing drawing: a pylon wing's tip device -- none draws no plate; vertical hangs "
        "straight, canted leans out, blended turns out of the wing -- each continuing the "
        "wing's chord",
        tips["none"][0] == 0 and all(tips[k_][0] > 0 for k_ in ("vertical", "canted", "blended"))
        and abs(float(fv[1]) - 0.708) < 1e-9
        and abs(float(fc[1]) - 0.708 - 0.12 * math.cos(math.radians(70.0))) < 1e-9
        and abs(float(fb[1]) - 0.708 - bl.projection(0.12, 90.0, 0.5, "spiral")) < 1e-9
        and abs(crb - c_tip) < 1e-9 and cfv < c_tip and cfb < c_tip,
        f"plate polys none {tips['none'][0]}, vertical {tips['vertical'][0]}, canted "
        f"{tips['canted'][0]}, blended {tips['blended'][0]}; far end out by 0 / "
        f"{float(fc[1]) - 0.708:.3f} / {float(fb[1]) - 0.708:.3f} m; far chords {cfv:.3f} / "
        f"{cfc:.3f} / {cfb:.3f} m under the {c_tip:.3f} m tip")

    # -- 6. a wing with the defaults draws exactly its old boxes ------------
    def corners(polys):
        return {tuple(np.round(p_, 9)) for v in polys for p_ in v}

    def same(a, b):                # the same corners, to 1e-9 m (a turned set's
        ua = np.unique(np.round(np.vstack(a), 9), axis=0)   # last bits differ)
        ub = np.unique(np.round(np.vstack(b), 9), axis=0)
        return len(ua) == len(ub) and all(np.min(np.linalg.norm(ub - q_, axis=1)) < 1e-9 for q_ in ua)
    ok_d = True
    for dep in (0.0, 0.6, 1.0):
        sp = spec_of("top", taper=0.7)
        zc = (deck_z(-0.90) + TOP_STOW_GAP) + (1.55 - deck_z(-0.90) - TOP_STOW_GAP) * dep
        old = []
        for sgn in (-1.0, 1.0):
            y = sgn * (0.70 + 0.008)
            old += [v for v, _c, _k in _box(-0.90 - 0.65 * 0.21, -0.90 + 0.65 * 0.21, y - 0.006,
                                             y + 0.006, zc - 0.12 * dep - 0.02 * (1 - dep),
                                             zc + 0.03, C_PLATE, "plate")]
        #  task 46: ...turned with the tip section, the wing's 6 deg x deploy,
        #  about the tip's mid-chord -- the section's own rotation
        a_ = math.radians(6.0) * dep
        ca_, sa_ = math.cos(a_), math.sin(a_)
        old = [np.array([(-0.90 + (p_[0] + 0.90) * ca_ + (p_[2] - zc) * sa_, p_[1],
                          zc - (p_[0] + 0.90) * sa_ + (p_[2] - zc) * ca_) for p_ in v]) for v in old]
        new = parts(sp, "top", dep)["plate"]
        ok_d &= len(new) == 12 and same(new, old)
        sf = spec_of("flank", taper=0.8)
        yc = CAR_HALF_W + DEV_OUT0 + DEV_OUT1 * dep
        old = []
        for sgn in (-1.0, 1.0):
            z = 0.90 + sgn * 0.39
            old += [v for v, _c, _k in _box(0.97 - 0.6 * 0.36, 0.97 + 0.6 * 0.36, yc - 0.06, yc + 0.06,
                                             z - 0.006, z + 0.006, C_PLATE, "plate")]
        #  the lower strut is the old box (the side is there at its height);
        #  the upper one, over the bonnet's edge, braces down onto it (task 46)
        dz_ = 0.28 * 0.78
        old += [v for v, _c, _k in _box(0.955, 0.985, CAR_HALF_W, yc, 0.90 - dz_ - 0.012,
                                         0.90 - dz_ + 0.012, C_STRUT)]
        up_b = side_land(None, 0.97, 0.90 + dz_)
        ok_d &= up_b[1] < 0.90 + dz_ - 0.1
        old += [v for v, _c, _k in _beam((0.97, up_b[0], up_b[1]), (0.97, yc, 0.90 + dz_),
                                         0.015, 0.012, C_STRUT)]
        pf_ = parts(sf, "left", dep)
        ok_d &= len(pf_["plate"]) + len(pf_["strut"]) == 24 \
            and same(pf_["plate"] + pf_["strut"], old)
    rep("wing drawing: a wing with the default mount rows draws its plates (and a flank "
        "panel its struts) exactly as before, deployed, half-way and stowed",
        ok_d, "top plates 1.3 x tip chord (turned with the tip, task 46), flank plates 1.2 x "
              "across the tip, struts at +-0.28 of the span (the upper one braced onto the "
              "bonnet's edge); only the top wing's pylons became swan necks")

    # -- 7. polygon counts stay bounded (the game draws these every frame) --
    heavy = dict(mount="pylon", pylon_frac=0.35, taper=0.6, plate_cant_deg=70.0,
                 plate_blend=0.4, plate_shape="spiral", plate_chord_follows=True,
                 plate_chord_ratio=1.8)
    ht, hf = parts(spec_of("top", **heavy), "top"), parts(spec_of("flank", **heavy), "left")
    n_t = sum(len(v) for v in ht.values())
    n_f = sum(len(v) for v in hf.values())
    rep("wing drawing: polygon counts stay bounded with every row on",
        len(ht["plate"]) <= 60 and len(hf["plate"]) <= 60 and len(ht["strut"]) <= 60
        and n_t + 2 * n_f <= 1.5 * 398,
        f"top {n_t} ({len(ht['plate'])} plate, {len(ht['strut'])} pylon), flank {n_f} "
        f"a side; the three {n_t + 2 * n_f} (398 before the mount was drawn; bound 1.5 x)")

    # -- 7b. AeroBO's own plate, and carsim's bracket to THIS body ----------
    #  (the owner saw a leaning plate drawn half a metre down the car's side:
    #  carsim's reach passed off as AeroBO's part). A carrying plate AeroBO
    #  designed is drawn to its own arc; where that stops short of the drawn
    #  body a slim dark bracket carries on along its line and lands ON it
    lean = dict(mount="endplate", plate_h=0.20, plate_cant_deg=70.0, plate_blend=0.4,
                plate_shape="spiral", design={"engine": "aerobo"})
    pa = parts(spec_of("top", **lean), "top")
    pc = parts(spec_of("top", **{k: v for k, v in lean.items() if k != "design"}), "top")
    pl = parts(spec_of("top", **dict(lean, plate_h=1.5)), "top")
    st_ = bl.plate_stations(0.20, 70.0, 0.4, "spiral")
    want = math.hypot(float(st_["dy"][-1]), float(st_["dz"][-1]))
    far = max(math.hypot(float(p_[1]) - 0.708, float(p_[2]) - 1.55)
              for v in pa["plate"] for p_ in v if p_[1] > 0.0)
    foot = [p_ for v in pa["bracket"] for p_ in v if p_[1] > 0.0]
    foot = sorted(foot, key=lambda p_: p_[2])[:4]
    on_body = foot and max(abs(float(p_[2]) - _deck_top(None, float(p_[0]))(float(p_[1])))
                           for p_ in foot) < 1e-9
    rep("wing drawing: an WingLab carrying plate is drawn to its OWN arc and bracketed to the "
        "body; a carsim carrying plate still runs to the body; a long enough one has no bracket",
        abs(far - want) < 0.012 and len(pa["bracket"]) > 0 and on_body
        and len(pc["bracket"]) == 0 and len(pl["bracket"]) == 0,
        f"plate end {far:.3f} m from the junction (its 0.20 m arc: {want:.3f}); "
        f"{len(pa['bracket'])} bracket polys, foot on the body {bool(on_body)}; carsim "
        f"{len(pc['bracket'])}, a 1.5 m plate {len(pl['bracket'])}")

    # -- 7c. the owner, 2026-09-27: "Wing tips not well connected" -----------
    #  WingLab's blend-1 flank (their own run: a 0.53 m wing, 0.45 m plates
    #  flown 0.25 m off the side at h 0.69) deploys to its flown standoff --
    #  slid out 0.35 m more, its rigid plates ended in mid-air -- so both
    #  plates' feet are on the car's side line, and the upper one, above the
    #  belt at the front wheels, is stayed down onto it. Stowed (the owner,
    #  same day: switched off, plates that carry the wing "don't disappear")
    #  it retracts against the side and its plates run on into the body; the
    #  plates are the wing's own colour either way
    ow_slot = Slot("f", 0.97, 0.69, 6.0)
    ow = spec_of("flank", mount="endplate", span=0.528, chord=0.204, taper=0.70, plate_h=0.4535,
                 plate_blend=1.0, plate_shape="spiral", plate_chord_ratio=1.7575, ride_h=0.25,
                 design={"engine": "aerobo"})
    belt = _deck_top(None, 0.97)(CAR_HALF_W)
    stow = flank_out(ow, 0.0, ow_slot.inc_deg)
    bad = []
    for dep, want in ((0.0, stow), (1.0, 0.25)):
        pp = {"plate": [], "wing": [], "bracket": []}
        cols = {"plate": set(), "wing": set()}
        for v, c_, kd in wing_polys(ow, "left", ow_slot, dep, lib):
            pp.setdefault(kd, []).append(np.asarray(v, float))
            cols.setdefault(kd, set()).add(tuple(c_))
        wy = np.vstack(pp["wing"])[:, 1].mean()
        py = np.vstack(pp["plate"])[:, 1]
        stay = np.vstack(pp["bracket"]) if pp["bracket"] else np.zeros((0, 3))
        if abs(wy - (CAR_HALF_W + want)) > 0.03:
            bad.append(f"deploy {dep:g}: the wing stands {wy - CAR_HALF_W:.3f} m off the side "
                       f"(want {want:.3f})")
        #  (task 46: a foot is set down on the side at its own height -- low
        #  down, the sill's tuck to 0.92 of the widest line)
        if not (0.92 * CAR_HALF_W - 0.01 <= py.min() <= CAR_HALF_W + 0.01) \
                or py.max() > CAR_HALF_W + want + 0.03:
            bad.append(f"deploy {dep:g}: the plates run {py.min() - CAR_HALF_W:.3f}.."
                       f"{py.max() - CAR_HALF_W:.3f} m off the side")
        if cols["plate"] != cols["wing"]:
            bad.append(f"deploy {dep:g}: plates {sorted(cols['plate'])}, wing {sorted(cols['wing'])}")
        if dep > 0.0 and (len(pp["bracket"]) != 6 or abs(stay[:, 2].min() - belt) > 1e-9):
            bad.append(f"deploy {dep:g}: {len(pp['bracket'])} stay polys, foot "
                       f"{stay[:, 2].min() if len(stay) else float('nan'):.3f} (belt {belt:.3f})")
    carsim_ep = spec_of("flank", mount="endplate")
    slide = (flank_out(spec_of("flank"), 0.0), flank_out(spec_of("flank"), 1.0),
             flank_out(carsim_ep, 0.0), flank_out(carsim_ep, 1.0))
    rep("wing drawing: a flank its endplates carry deploys to its flown standoff (plate feet on "
        "the side, the one above the belt stayed onto it) and stowed retracts against the side, "
        "plates into the body; plates in the wing's colour; a pylon wing still slides out",
        not bad and stow < 0.10 and abs(slide[0] - DEV_OUT0) < 1e-12
        and abs(slide[1] - (DEV_OUT0 + DEV_OUT1)) < 1e-12 and slide[2] < 0.15
        and abs(slide[3] - (DEV_OUT0 + DEV_OUT1)) < 1e-12,
        "; ".join(bad) if bad else
        f"WingLab wing {stow:.3f} m off the side stowed, 0.25 deployed, plate feet on the side, "
        f"one stay down to the belt at {belt:.3f} m; pylon {slide[0]:.2f} -> {slide[1]:.2f} m, "
        f"carsim endplates {slide[2]:.3f} -> {slide[3]:.2f} m")

    # -- 7d. task 46: a tip device caps its tip, struts and pylons stand on
    #  the car (the owner, 2026-09-28: "Wing tip not connected properly to
    #  wing in pylon"; "when pushing the wing forward it sometimes is floating
    #  in the air")
    bad46 = []
    for role, key, slot_ in (("flank", "left", Slot("f", 0.97, 0.90, 12.0)),
                             ("top", "top", Slot("t", -0.90, 1.55, 12.0, "active"))):
        for tw_ in (0.0, 3.0):
            for dev in ({}, dict(plate_cant_deg=75.0), dict(plate_blend=0.5, plate_shape="spiral")):
                sp = spec_of(role, mount="pylon", pylon_frac=0.35, taper=0.6, twist_deg=tw_,
                             plate_chord_follows=True, plate_h=0.10, design={"engine": "aerobo"},
                             **dev)
                pw = wing_polys(sp, key, slot_, 1.0, lib)
                if "plate_blend" in dev:
                    #  a blend grows out of the wing, tangent to it: its first
                    #  ring lies along the tip's own chord line
                    err = tip_turn_error(pw, role)
                    if err > 0.5:
                        bad46.append(f"{role} twist {tw_:g} blended: its root {err:.1f} deg off "
                                     f"the tip chord")
                    continue
                cov = tip_cover(pw, role)
                if cov < 1.0:
                    bad46.append(f"{role} twist {tw_:g} {dev or 'vertical'}: {100 * cov:.0f} % of "
                                 f"the tip section on its plate")
    rep("wing drawing (task 46): a pylon wing's tip device caps its whole tip section, "
        "turned with the wing's incidence and twist (vertical, canted, blended; 12 deg)",
        not bad46, "; ".join(bad46) if bad46 else "every tip-section point on its plate, 12 cases")
    #  every flank strut's body end is ON the drawn body -- the mesh sliced at
    #  its station -- over the whole slot band, on the Corsa and the Express
    worst, n_st, braced = 0.0, 0, 0
    for car_ in ("corsa", "express"):
        geo_ = PreviewGeo(car_)
        mesh_ = Batch(build_car_mesh(car=car_))
        x_lo_, x_hi_ = bodies.flank_x_band(car_, 0.45)
        h_lo_, h_hi_ = bodies.flank_h_band(car_)
        sp = spec_of("flank", mount="pylon", pylon_frac=0.56)
        for x_ in np.linspace(x_lo_, x_hi_, 7):
            sec_ = _mesh_section(mesh_, float(x_))
            for h_ in np.linspace(h_lo_, h_hi_, 5):
                sv = [np.asarray(v, float) for v, _c, kd in
                      wing_polys(sp, "left", Slot("f", float(x_), float(h_), 0.0), 1.0, lib, geo=geo_)
                      if kd == "strut"]
                for i_ in range(0, len(sv), 6):
                    q_ = np.vstack(sv[i_:i_ + 6])
                    end = q_[np.abs(q_[:, 1]) <= np.abs(q_[:, 1]).min() + 0.015]
                    y_, z_ = float(np.abs(end[:, 1]).mean()), float(end[:, 2].mean())
                    worst = max(worst, _seg_dist2(sec_, y_, z_))
                    n_st += 1
                    braced += int(np.ptp(q_[:, 2]) > 0.03)
    rep("wing drawing (task 46): every side-wing strut lands ON the drawn body -- level "
        "where it has a side there, braced to its top edge (a bonnet, a tail) where not",
        worst < 0.02 and braced > 0,
        f"{n_st} struts over both slot bands (Corsa, Express): body end at most "
        f"{worst * 1000:.1f} mm off the drawn section; {braced} braced")
    #  the top wing held where what carries it stands on the car
    rows46 = []
    ok46 = True
    for car_ in ("corsa", "express"):
        for m_ in ("pylon", "endplate"):
            lib_t = type("L", (), {})()
            sp = spec_of("top", mount=m_, pylon_frac=0.56, chord=0.40)
            lib_t.wings = {"t46": sp}
            b_ = CarBuild.for_car(car_)
            b_.top.wing, b_.top.x = "t46", -99.0
            b_.clamp(lib_t, car_)
            geo_ = PreviewGeo(car_)
            pv = [np.asarray(v, float) for v, _c, kd in
                  wing_polys(sp, "top", b_.top, 1.0, lib, geo=geo_) if kd in ("strut", "plate")]
            x_min = float(np.vstack(pv)[:, 0].min())
            ok46 &= x_min >= geo_.x_rear - 1e-9 and abs(b_.top.x - top_x_floor(car_, sp)) < 1e-12
            rows46.append(f"{car_} {m_}: x {b_.top.x:+.3f}, rearmost part {x_min - geo_.x_rear:+.3f} m "
                          f"inside the tail")
    rep("wing drawing (task 46): the top wing goes no further back than where its pylons' "
        "feet (or its endplates) stand on the car", ok46, "; ".join(rows46))

    # -- 8. the pictures ----------------------------------------------------
    if view is not None and screenshot_dir:
        os.makedirs(screenshot_dir, exist_ok=True)
        if lib is None:
            import tempfile
            lib = Library(tempfile.mkdtemp(prefix="carsim_draw_"), use_xfoil=False)
        tmp_lib = lib
        cases = (("endplate_upright", dict(mount="endplate")),
                 ("endplate_lean70_blend04_ratio18",
                  dict(mount="endplate", plate_cant_deg=70.0, plate_blend=0.4,
                       plate_shape="spiral", plate_chord_ratio=1.8)),
                 ("pylon_canted70_follows", dict(mount="pylon", pylon_frac=0.35, taper=0.6,
                                                 plate_cant_deg=70.0, plate_chord_follows=True)),
                 ("aerobo_endplate_lean70_blend04", dict(lean, plate_h=0.25)))
        cam = Orbit(view.W, view.H)
        saved = []
        for tag, kw in cases:
            bld = CarBuild()
            for key, role in (("top", "top"), ("left", "flank")):
                name = f"_draw_{tag}_{role}"
                tmp_lib.wings[name] = spec_of(role, name=name, **kw)
                bld.slot(key).wing = name
            bld.right = Slot(bld.left.wing, bld.left.x, bld.left.h, bld.left.inc_deg)
            bld.top.h = 1.80                    # clear of the roof: the mount shows
            for view_tag, yaw, pitch, dist, tgt in (("top", 212.0, 16.0, 3.0, (-0.9, 0.0, 1.55)),
                                                    ("flank", 40.0, 10.0, 3.0, (0.97, 0.8, 0.9))):
                cam.yaw, cam.pitch = math.radians(yaw), math.radians(pitch)
                cam.dist, cam.target = dist, np.array(tgt)
                view.draw_scene(bld, tmp_lib, cam, 1.0, "")
                path = os.path.join(screenshot_dir, f"garage_mount_{tag}_{view_tag}.png")
                pygame.image.save(view.screen, path)
                saved.append(path)
            for key in ("top", "left"):
                tmp_lib.wings.pop(bld.slot(key).wing, None)
        rep("wing drawing: the car page's preview of the three mounts saved",
            all(os.path.exists(p_) for p_ in saved), ", ".join(os.path.basename(p_) for p_ in saved))
    return rows


def panel_mesh(design: WingDesign, deploy: float = 0.0):
    """The published panel on both flanks (kept for the old callers)."""
    slot = Slot(design.wing if design.wing != "off" else "", design.x_w, design.h_w, design.inc_deg)
    out = []
    for key in ("left", "right"):
        out += wing_polys(None, key, slot, deploy, None, legacy_type=(design.wing if design.wing != "off" else "fin"))
    return out


# =========================================================================== #
#  CAMERA                                                                      #
# =========================================================================== #
class Orbit:
    """Yaw / pitch / distance about a target; perspective projection."""

    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        #  the car being framed (task 41): 1 and the Corsa's centre unless
        #  `fit` says otherwise -- a 12 m bus stands the camera ~3x back
        self.scale = 1.0
        self.target0 = (-0.15, 0.0, 0.70)
        self.reset()
        self.fov_deg = 38.0

    def fit(self, geo: "PreviewGeo") -> None:
        """Frame `geo`'s car: its centre, and the distance (and the zoom's
        range) scaled to its size. The Corsa's view is unchanged."""
        self.scale, self.target0 = float(geo.scale), tuple(geo.target)
        self.reset()

    def reset(self) -> None:
        self.yaw = math.radians(38.0)        # from the front-left quarter
        self.pitch = math.radians(19.0)
        self.dist = 7.6 * self.scale
        self.target = np.array(self.target0)

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return self.target + self.dist * np.array(
            [cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)])

    def basis(self):
        eye = self.eye()
        f = self.target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return eye, r, u, f

    def project(self, P: np.ndarray):
        """(n,3) world -> (n,2) screen, (n,) depth. Depth <= 0.05 is behind."""
        eye, r, u, f = self.basis()
        d = np.asarray(P, dtype=np.float64) - eye
        px, py, pz = d @ r, d @ u, d @ f
        fl = 0.5 * self.H / math.tan(0.5 * math.radians(self.fov_deg))
        pz_safe = np.where(pz > 0.05, pz, 0.05)
        sx = 0.5 * self.W + fl * px / pz_safe
        sy = 0.5 * self.H - fl * py / pz_safe
        return np.stack([sx, sy], axis=1), pz

    def orbit(self, dyaw: float, dpitch: float) -> None:
        self.yaw += dyaw
        self.pitch = min(max(self.pitch + dpitch, math.radians(-5.0)),
                         math.radians(80.0))

    def zoom(self, k: float) -> None:
        self.dist = min(max(self.dist * k, 3.2 * self.scale), 16.0 * self.scale)


# =========================================================================== #
#  THE 3-D VIEW (the CAR page)                                                 #
# =========================================================================== #
def car_spec(car=None):
    """`cars.py`'s spec of `car` for the page's read-outs (%mg, (x + b)/b,
    the front / rear split): the Corsa's for None, and for a key `cars.py`
    does not carry yet (bodies draws its shell from the style car alone)."""
    import cars
    return cars.CARS.get(car or "corsa") or cars.CARS["corsa"]


def limit_rows(build: "CarBuild", lib: "Library", car=None) -> list:
    """Each FITTED wing's span against its physical limit on `car` (task 41;
    `bodies.span_limit`): [(slot, span, limit, past)] in slot order -- what
    the car page's SPAN LIMITS panel lists. `past` is `bodies.over_limits`'
    own test, so the panel and a run's Unlimited flag cannot disagree."""
    over = {o["slot"] for o in bodies.over_limits(build, lib, car)}
    out = []
    for key in SLOTS:
        slot = build.slot(key)
        spec = lib.wings.get(slot.wing) if slot.wing else None
        if spec is None:
            continue
        out.append((key, float(spec.span), bodies.span_limit(SLOT_ROLE[key], car, slot.h),
                    key in over))
    return out


class GarageView:
    def __init__(self, screen: pygame.Surface, car: str = "corsa"):
        self.screen = screen
        self.W, self.H = screen.get_size()
        self.ui = min(self.W / 1280.0, self.H / 800.0)
        self.fonts = ui.Fonts(self.ui)
        self.text = ui.Text(self.fonts)
        self.f_lbl = self.fonts.get(14)
        self.f_val = self.fonts.get(16)
        self.f_big = self.fonts.get(26, bold=True)
        self.paint = C_PAINT
        #: task 41: the car being fitted -- its key, what the preview draws
        #: round, and its spec for the read-outs
        self.car_key = car or "corsa"
        self.geo = PreviewGeo(self.car_key)
        self.spec = car_spec(self.car_key)
        self.car = Batch(build_car_mesh(self.paint, self.car_key))
        self.n_polys = 0
        self.frame_ms = 0.0
        #: task 45: the force arrows start OFF (their labels piled over the
        #: wings on a first look); V shows them
        self.show_vectors = False
        #: task 45: the slot panel's lines and the arrows' legend as the last
        #: frame drew them -- the self-check reads both
        self.info_drawn: list = []
        self.legend_drawn = ""
        #: task 45: where the vectors' legend ends (px; 0 when it is not
        #: drawn), and the hint's lines as the last frame drew them,
        #: [(text, x, y)] -- the hint keeps clear of the one, the self-check
        #: reads the other
        self._legend_r = 0.0
        self.hint_drawn: list = []
        #: task 45: the BUILD line as the last frame drew it, (text, rect)
        self.header_drawn: tuple = ("", None)
        #: task 45: the SPAN LIMITS panel's head as the last frame drew it,
        #: (the car's line, the setting) -- the self-check reads it
        self.limits_drawn: tuple = ("", "")

    def set_paint(self, rgb) -> None:
        """Paint the preview car `rgb` (None: the stock C_PAINT yellow). The
        body Batch is rebuilt only when the colour changes (~2.4 ms); the next
        frame draws it. Cosmetic: nothing in the build or its JSON changes.
        The caller passes the fitted car's paint RESOLVED (its factory
        colour for 'factory', render.factory_colour); the preview is that
        car's own body since task 41."""
        rgb = C_PAINT if rgb is None else tuple(int(c) for c in rgb)
        if rgb != self.paint:
            self.paint = rgb
            self.car = Batch(build_car_mesh(rgb, self.car_key))

    def _txt(self, s, x, y, font=None, col=C_TEXT, alpha: float = 1.0):
        surf = (font or self.f_val).render(s, True, col)
        if alpha < 1.0:
            surf.set_alpha(int(255 * max(0.0, alpha)))
        self.screen.blit(surf, (int(x), int(y)))
        return surf.get_width()

    # ---------------------------------------------------------------- #
    def draw_scene(self, build: CarBuild, lib: Library, cam: Orbit, deploy: float,
                   selected: str) -> None:
        sc = self.screen
        sc.fill(C_BG)
        self._draw_ground(cam)
        eye = cam.eye()
        wings = build.wings(lib)
        polys = []
        for key in SLOTS:
            polys += wing_polys(wings[key], key, build.slot(key), deploy, lib,
                                selected=(key == selected), geo=self.geo)
        b = Batch.join(self.car, Batch(polys)) if polys else self.car
        view = eye - b.centroids
        vlen = np.linalg.norm(view, axis=1, keepdims=True)
        vdir = view / np.where(vlen > 1e-9, vlen, 1e-9)
        facing = np.einsum("ij,ij->i", b.normals, vdir) > 0.0
        keep = facing | b.no_cull
        scr, depth = cam.project(b.verts)
        lam = np.clip(np.abs(b.normals @ LIGHT_DIR), 0.0, None)
        half = (LIGHT_DIR[None, :] + vdir) * 0.5
        spec = np.clip(np.abs(np.einsum("ij,ij->i", b.normals, half)), 0.0, None) ** 8
        gain = 0.42 + 0.50 * lam
        d_min = np.minimum.reduceat(depth, b.starts)
        d_mean = np.add.reduceat(depth, b.starts) / b.counts
        vis = keep & (d_min > 0.05)
        idx = np.nonzero(vis)[0]
        order = idx[np.argsort(-d_mean[idx], kind="stable")]
        self.n_polys = int(order.size)
        scr_i = scr.astype(np.int32)
        starts, counts = b.starts, b.counts
        for i in order:
            s0, c0 = starts[i], counts[i]
            col = tuple(min(255, int(c * gain[i] + 60.0 * spec[i])) for c in b.colours[i])
            pts = scr_i[s0:s0 + c0].tolist()
            if len(pts) < 3:
                continue
            pygame.draw.polygon(sc, col, pts)
            if b.kinds[i] == "wingsel":
                pygame.draw.polygon(sc, (255, 236, 200), pts, 1)

    def draw(self, build: CarBuild, lib: Library, cam: Orbit, deploy: float, selected: str,
             pad_name: str | None, hint: str = "", hint_alpha: float = 1.0,
             unlimited: bool = False, avoid=None, header=None) -> None:
        """The car page. `hint_alpha` is how much of the hint is left (the
        garage fades it, task 45); `avoid` a box drawn over the page's lower
        right afterwards (the wing tutorial's) that the hint keeps left of;
        `header` the BUILD line (`Garage.build_header`), None for none."""
        t0 = time.perf_counter()
        self.draw_scene(build, lib, cam, deploy, selected)
        self._legend_r, self.legend_drawn = 0.0, ""
        if self.show_vectors:
            self._draw_vectors(build, lib, cam, deploy)
        self._draw_dimensions(build, selected, cam)
        self._draw_header(header)
        self._draw_info(build, lib, selected, deploy)
        self._draw_limits(build, lib, unlimited)
        self._draw_help(pad_name, hint, hint_alpha, avoid)
        self.frame_ms = (time.perf_counter() - t0) * 1e3

    def _draw_header(self, header) -> None:
        """The BUILD line over the page's top left (task 45): `header`'s
        [(text, colour)] on a panel of its own, left of the slot panel and
        SPAN LIMITS (x 884) -- the name, the second part, cut to fit when
        the line would reach them. `header_drawn` keeps (text, rect)."""
        self.header_drawn = ("", None)
        if not header:
            return
        u, f, fn = self.ui, self.f_lbl, self.f_val
        pad, room = 12 * u, (884 - 12 - 12) * u
        segs = [list(s_) for s_ in header]
        if len(segs) > 1:
            rest = sum(f.size(s_)[0] for i, (s_, _) in enumerate(segs) if i != 1)
            segs[1][0] = self._fit(segs[1][0], fn, max(60 * u, room - 2 * pad - rest))
        wide = sum((fn if i == 1 else f).size(s_)[0] for i, (s_, _) in enumerate(segs))
        r = self._panel((12, 12, wide / u + 24, 30))
        x, mid = r.x + pad, r.centery
        for i, (s_, col) in enumerate(segs):
            font = fn if i == 1 else f
            x += self._txt(s_, x, mid - font.get_height() // 2, font, col)
        self.header_drawn = ("".join(s_ for s_, _ in segs), r)

    # ---------------------------------------------------------------- #
    def _line3(self, cam, a, b, col, width=1):
        scr, d = cam.project(np.array([a, b]))
        if np.any(d <= 0.05):
            return None
        p = scr.astype(np.int32).tolist()
        pygame.draw.line(self.screen, col, p[0], p[1], width)
        return p

    # ---------------------------------------------------------------- #
    def _arrow3(self, cam, a, b, col, width=2):
        """A world-space arrow a -> b: line plus a screen-space head."""
        scr, d = cam.project(np.array([a, b]))
        if np.any(d <= 0.05):
            return None
        (x0, y0), (x1, y1) = scr.tolist()
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1.0:
            return None
        pygame.draw.line(self.screen, col, (int(x0), int(y0)), (int(x1), int(y1)), width)
        if L > 6.0:
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            h = min(9.0, 0.3 * L)
            pygame.draw.polygon(self.screen, col, [
                (int(x1), int(y1)),
                (int(x1 - h * (ux + 0.5 * uy)), int(y1 - h * (uy - 0.5 * ux))),
                (int(x1 - h * (ux - 0.5 * uy)), int(y1 - h * (uy + 0.5 * ux)))])
        return (x1, y1)

    def wing_vectors(self, build: CarBuild, lib: Library, deploy: float) -> list:
        """The force vectors of every fitted wing at its role's V_REF, in
        world metres: (key, base, tip, newtons, kind) with kind 'F' | 'D'.

        One scale, VEC_M_PER_N, for all three wings and both components,
        so the eye compares a flank panel's side force against the top
        wing's downforce honestly. Anchored where the wing IS at this deploy
        fraction (the same standoff / rise `wing_polys` draws), at the
        slot's station and height. A flank panel's force points INTO the
        car (its lift is the inward side force); the top wing's points DOWN
        (F_top > 0 is downforce); drag points rearward, -x."""
        out = []
        wings = build.wings(lib)
        for key in SLOTS:
            spec = wings[key]
            if spec is None or not (spec.aero or spec.legacy):
                continue
            slot = build.slot(key)
            dp = design_point(shown_spec(spec), slot.inc_deg, V=V_REF[spec.role], x_w=slot.x)
            if not dp:
                continue
            if key == "top":
                z_stow = self.geo.deck_z(slot.x) + TOP_STOW_GAP
                base = np.array([slot.x, 0.0, z_stow + (slot.h - z_stow) * deploy])
                f_dir = np.array([0.0, 0.0, -1.0])
            else:
                s = 1.0 if key == "left" else -1.0
                base = np.array([slot.x, s * (self.geo.half_w + flank_out(spec, deploy,
                                                                          slot.inc_deg)),
                                 slot.h])
                f_dir = np.array([0.0, -s, 0.0])
            d_dir = np.array([-1.0, 0.0, 0.0])
            for F, dv, kind in ((dp["F"], f_dir, "F"), (dp["D"], d_dir, "D")):
                if abs(F) < 0.5:
                    continue
                out.append((key, base, base + dv * (F * VEC_M_PER_N), float(F), kind))
        return out

    def _draw_vectors(self, build: CarBuild, lib: Library, cam: Orbit, deploy: float) -> None:
        vecs = self.wing_vectors(build, lib, deploy)
        for key, a, b, F, kind in vecs:
            col = C_VEC_F if kind == "F" else C_VEC_D
            tip = self._arrow3(cam, a, b, col)
            if tip is None:
                continue
            self._txt(f"{kind} {F:.0f} N", tip[0] + 6, tip[1] - 8, self.f_lbl, col)
        if vecs:
            u = self.ui
            #  task 45: the speed in km/h instead of the code's V_REF -- each
            #  role's own (a flank panel's 105 km/h, a top wing's 144)
            kmh = {r: f"{3.6 * V_REF[r]:.0f} km/h" for r in ("flank", "top")}
            roles = {SLOT_ROLE[k] for k, *_ in vecs}
            at = (f"at {kmh[roles.pop()]}" if len(roles) == 1
                  else f"(sides at {kmh['flank']}, top at {kmh['top']})")
            self.legend_drawn = (f"arrows: {1.0 / VEC_M_PER_N:.0f} N = 1 m {at}   "
                                 "F force   D drag   (V hides)")
            self._legend_r = 12 * u + self._txt(self.legend_drawn, 12 * u, 668 * u,
                                                self.f_lbl, C_TEXT_DIM)

    def _draw_ground(self, cam: Orbit) -> None:
        geo = self.geo
        k, xc = geo.scale, (0.0 if geo.corsa else geo.target[0])   # the Corsa's: 1, 0
        quad = np.array([(xc - 6.0 * k, -5.0 * k, 0.0), (xc + 6.0 * k, -5.0 * k, 0.0),
                         (xc + 6.0 * k, 5.0 * k, 0.0), (xc - 6.0 * k, 5.0 * k, 0.0)])
        scr, d = cam.project(quad)
        if np.all(d > 0.05):
            pygame.draw.polygon(self.screen, C_GROUND, scr.astype(np.int32).tolist())
        for x in np.arange(-4.0 * k, 4.0 * k + 0.01, 0.5):
            self._line3(cam, (xc + x, -3.0 * k, 0.0), (xc + x, 3.0 * k, 0.0), C_GRID)
        for y in np.arange(-3.0 * k, 3.0 * k + 0.01, 0.5):
            self._line3(cam, (xc - 4.0 * k, y, 0.0), (xc + 4.0 * k, y, 0.0), C_GRID)
        xf, xr = geo.x_front, geo.x_rear
        wa, wb, wr = ((0.62, 0.86, 0.66) if geo.corsa else
                      (0.75 * geo.half_w, 1.045 * geo.half_w, 0.80 * geo.half_w))
        foot = np.array([(xf - 0.05, wa, 0.003), (xf - 0.35, wb, 0.003),
                         (xr + 0.30, wb, 0.003), (xr, wr, 0.003),
                         (xr, -wr, 0.003), (xr + 0.30, -wb, 0.003),
                         (xf - 0.35, -wb, 0.003), (xf - 0.05, -wa, 0.003)])
        scr, d = cam.project(foot)
        if np.all(d > 0.05):
            pygame.draw.polygon(self.screen, C_SHADOW, scr.astype(np.int32).tolist())

    def _draw_dimensions(self, build: CarBuild, key: str, cam: Orbit) -> None:
        """CG marker, x along the ground, h up the side, for the selected slot."""
        slot = build.slot(key)
        s = -1.0 if key == "right" else 1.0
        y_side = s * (self.geo.half_w + DEV_OUT0 + 0.45)
        self._line3(cam, (-0.15, 0.0, 0.0), (0.15, 0.0, 0.0), C_DIM, 2)
        self._line3(cam, (0.0, -0.15, 0.0), (0.0, 0.15, 0.0), C_DIM, 2)
        p = cam.project(np.array([(0.0, 0.0, 0.0)]))[0][0]
        self._txt("CG", p[0] + 6, p[1] - 22, self.f_lbl, C_DIM)
        self._line3(cam, (0.0, y_side, 0.0), (slot.x, y_side, 0.0), C_DIM, 2)
        self._line3(cam, (0.0, y_side - 0.1, 0.0), (0.0, y_side + 0.1, 0.0), C_DIM, 2)
        self._line3(cam, (slot.x, y_side - 0.1, 0.0), (slot.x, y_side + 0.1, 0.0), C_DIM, 2)
        p = cam.project(np.array([(0.5 * slot.x, y_side + s * 0.15, 0.0)]))[0][0]
        self._txt(f"x {slot.x:+.2f} m", p[0] - 40, p[1] + 4, self.f_lbl, C_DIM)
        yp = (s * (self.geo.half_w + DEV_OUT0 + 0.5 * DEV_THICK + 0.10) if key != "top"
              else self.geo.top_dim_y)
        self._line3(cam, (slot.x, yp, 0.0), (slot.x, yp, slot.h), C_DIM, 2)
        p = cam.project(np.array([(slot.x, yp, slot.h)]))[0][0]
        self._txt(f"h {slot.h:.2f} m", p[0] + 8, p[1] - 10, self.f_lbl, C_DIM)

    def _panel(self, rect):
        r = pygame.Rect(*(int(v * self.ui) for v in rect))
        s = pygame.Surface(r.size, pygame.SRCALPHA)
        s.fill((*C_HUD_BG, 200))
        self.screen.blit(s, r.topleft)
        return r

    def _fit(self, s: str, font, w: float) -> str:
        """`s` cut to `w` px in `font`, ending '...' when it had to be."""
        if font.size(s)[0] <= w:
            return s
        while s and font.size(s + "...")[0] > w:
            s = s[:-1]
        return s.rstrip() + "..."

    def _wrap_px(self, s: str, font, w: float) -> list:
        """`s` as the lines that fit `w` px in `font`: itself when it fits
        (its column spacing kept), else broken at the spaces."""
        if font.size(s)[0] <= w:
            return [s]
        out, line = [], ""
        for word in s.split():
            trial = f"{line} {word}".strip()
            if line and font.size(trial)[0] > w:
                out.append(line)
                line = word
            else:
                line = trial
        if line:
            out.append(line)
        return [self._fit(ln, font, w) for ln in out]

    def _draw_info(self, build: CarBuild, lib: Library, key: str, deploy: float) -> None:
        u = self.ui
        r = self._panel((884, 12, 384, 470))
        x, y = r.x + 12 * u, r.y + 8 * u
        wmax = r.right - 12 * u - x

        def put(s, yy, font, col=C_TEXT) -> float:
            """`s` at `yy`, wrapped to the panel (and at any newline in it):
            the px its extra lines took."""
            lines = [ln for part in s.split("\n") for ln in self._wrap_px(part, font, wmax)]
            for i, ln in enumerate(lines):
                self._txt(ln, x, yy + i * font.get_linesize(), font, col)
            self.info_drawn += lines
            return (len(lines) - 1) * font.get_linesize()

        slot = build.slot(key)
        spec = lib.wings.get(slot.wing)
        on = spec is not None
        head = self._fit(SLOT_LABEL[key] + ("   (mirrored)" if build.mirror and key != "top" else ""),
                         self.f_lbl, wmax)
        #  a built-in by the name W's hint and the library give it (task 45)
        name = self._fit(wing_shown(spec.name, lib)[0] if on else "none", self.f_big, wmax)
        self.info_drawn = [head, name]
        self._txt(head, x, y, self.f_lbl, C_PANEL_ON if on else C_TEXT_DIM)
        y += 20 * u
        self._txt(name, x, y, self.f_big, C_PANEL_ON if on else C_TEXT_DIM)
        y += 40 * u
        dp = None
        if on and (spec.aero or spec.legacy):
            V = V_REF[spec.role]
            dp = design_point(shown_spec(spec), slot.inc_deg, V=V, x_w=slot.x)
        if on:
            geo = (f"{spec.airfoil}  b {spec.span:.2f} c {spec.chord:.2f} taper {spec.taper:.2f} "
                   f"S {spec.S:.2f} m2")
            if spec.legacy:
                geo = f"published panel S {spec.legacy.get('S', 0.35):.2f} m2 CL0 {spec.legacy['CL0']:.2f} L/D {spec.legacy['LD']:.1f}"
            rows = [(geo, C_TEXT_DIM)]
            if dp:
                #  task 45: what the wing does for the car, in a player's
                #  words, ahead of the engineering rows
                top_, why, col = slot_summary(spec, key, dp)
                f_sum = self.fonts.get(14, bold=True)
                lines = [""]                     # too wide: broken at a ' · '
                for part in top_.split(" · "):
                    trial = f"{lines[-1]} · {part}" if lines[-1] else part
                    if lines[-1] and f_sum.size(trial)[0] > wmax:
                        lines.append(part)
                    else:
                        lines[-1] = trial
                y += put("\n".join(lines), y, f_sum, col)
                y += 19 * u
                y += put(why, y, self.f_lbl, C_TEXT_DIM)
                y += 24 * u
        else:
            rows = ([("Start here: W puts a ready-made wing\nin this slot", C_PANEL_ON)]
                    if key != "top" else [])
            rows += [("W  try a ready-made wing", C_TEXT_DIM), ("D  design your own", C_TEXT_DIM)]
        rows += [
            (f"station x    {slot.x:+.2f} m", C_TEXT),
            (f"             {station_label(slot.x, self.car_key)}", C_TEXT_DIM),
            (f"height  h    {slot.h:.2f} m", C_TEXT),
            (f"incidence    {slot.inc_deg:+.0f} deg", C_TEXT),
        ]
        if key == "top":
            rows.append((f"deploys      {'brake + steer (active)' if slot.mode == 'active' else 'whenever armed (fixed)'}", C_TEXT))
        for s_, c in rows:
            y += put(s_, y, self.f_val if c is C_TEXT else self.f_lbl, c)
            y += 22 * u
        y += 6 * u
        if on and (spec.aero or spec.legacy):
            if dp:
                if spec.engine == "aerobo":
                    tag = "  (WingLab's law)"            # sampled from AeroBO's evaluator
                else:
                    tag = "" if (spec.legacy or not spec.aero.get("polar_is_estimate", True)) else "  (estimated data)"
                y += put(f"at {3.6 * V:.0f} km/h{tag}", y, self.f_lbl, C_TEXT_DIM)
                y += 20 * u
                y += put(f"CL {dp['CL']:.2f}  F {dp['F']:4.0f} N  D {dp['D']:3.0f} N  L/D {dp['LD']:.1f}",
                         y, self.f_val)
                y += 22 * u
                car = self.spec                  # the fitted car's (task 41)
                if spec.role == "flank":
                    y += put(f"= {100 * dp['F'] / (car.m * G):.2f}% of mg  (x+b)/b x{(slot.x + car.b) / car.b:.2f}",
                             y, self.f_lbl, C_TEXT_DIM)
                    y += 22 * u
                    y += put("corner-speed gain", y, self.f_lbl, C_TEXT_DIM)
                    y += 20 * u
                    k = dp.get("k", 0.0)
                    for R in (50.0, 100.0, 130.0):
                        g = crossover.gain(k, R, slot.x)
                        if g is None:
                            s_, c = f"  R={R:3.0f} m   runaway", C_WARN
                        else:
                            gp = 100.0 * g
                            over = gp > GAIN_CAP_PCT
                            s_ = f"  R={R:3.0f} m   {gp:+6.2f}%" + ("   > cap: REAR-ltd" if over else "")
                            c = C_WARN if over else (C_OK if gp > 0 else C_TEXT_DIM)
                        y += put(s_, y, self.f_val, c)
                        y += 22 * u
                else:
                    split, share_f = split_text(slot.x, car)
                    y += put(f"downforce split  {split}", y, self.f_lbl,
                             C_OK if share_f > 0.3 else C_WARN)
                    y += 22 * u
                    y += put(f"= {100 * dp['F'] / (car.m * G):.2f}% of mg; a front-limited car wants it forward",
                             y, self.f_lbl, C_TEXT_DIM)
                    y += 22 * u
                if dp.get("stalled"):
                    y += put("STALLED at this incidence", y, self.f_val, C_WARN)
                    y += 22 * u
                else:
                    y += put(f"stall margin {dp['stall_margin_deg']:.1f} deg", y, self.f_lbl,
                             C_OK if dp["stall_margin_deg"] > 2.0 else C_WARN)
                    y += 22 * u
        y = r.bottom - 26 * u
        foot = self._fit(f"preview: {'DEPLOYED' if deploy > 0.5 else 'stowed'}   {build.summary(lib)}",
                         self.f_lbl, wmax)
        self.info_drawn.append(foot)
        self._txt(foot, x, y, self.f_lbl, C_PANEL_ON if deploy > 0.5 else C_TEXT_DIM)

    def _draw_limits(self, build: CarBuild, lib: Library, unlimited: bool) -> None:
        """SPAN LIMITS, under the slot panel (task 41): every fitted wing's
        span against its physical limit on THIS car -- a flank's lower tip
        at the car's ground clearance, a top wing 1.2 x its width -- PAST THE
        LIMIT in red, and what that means for a run: in Real mode a build
        that is past (a bus build on a Corsa) is kept and the page says its
        runs count as Unlimited; in Unlimited mode it wears the tag."""
        rows = limit_rows(build, lib, self.car_key)
        self.limits_drawn = ("", "")
        if not rows:
            return
        past = any(r[3] for r in rows)
        u = self.ui
        lines = 1 + len(rows) + (2 if past else 0)
        r = self._panel((884, 490, 384, 10 + 19 * lines))
        x, y = r.x + 12 * u, r.y + 6 * u
        wmax = r.right - 12 * u - x
        #  the SETTING in mixed case; the capital UNLIMITED is kept for the tag
        #  a build past its limit wears (the line under the rows). Task 45:
        #  the setting drops its "limits: " before the car's name is cut (it
        #  read 'Opel Corsa...' beside 'limits: Unlimited')
        mode = "limits: Unlimited" if unlimited else "limits: Real"
        import cars
        who = cars.CAR_TITLES.get(self.car_key, self.car_key)
        if self.f_lbl.size(f"SPAN LIMITS  {who}")[0] > wmax - self.f_lbl.size(mode)[0] - 8 * u:
            mode = mode[len("limits: "):]
        head = self._fit(f"SPAN LIMITS  {who}", self.f_lbl, wmax - self.f_lbl.size(mode)[0] - 8 * u)
        self.limits_drawn = (head, mode)
        self._txt(head, x, y, self.f_lbl, C_TEXT_DIM)
        self._txt(mode, r.right - 12 * u - self.f_lbl.size(mode)[0], y, self.f_lbl,
                  C_PANEL_ON if unlimited else C_TEXT_DIM)
        for key, span, lim, over in rows:
            y += 19 * u
            row = f"{key:<6s}span {span:.2f} m / max {lim:.2f} m"
            if over and self.f_lbl.size(row + "  PAST THE LIMIT")[0] > wmax:
                row = f"{key:<6s}span {span:.2f} / max {lim:.2f} m"
            self._txt(row, x, y, self.f_lbl, C_WARN if over else C_TEXT)
            if over:
                mark = "PAST THE LIMIT"
                if self.f_lbl.size(f"{row}  {mark}")[0] > wmax:
                    mark = "PAST"
                self._txt(mark, r.right - 12 * u - self.f_lbl.size(mark)[0], y, self.f_lbl, C_WARN)
        if past:
            y += 19 * u
            self._txt(self._fit("UNLIMITED: runs are filed apart" if unlimited
                                else "runs with this build count as UNLIMITED", self.f_lbl, wmax),
                      x, y, self.f_lbl, C_WARN)
            y += 19 * u
            self._txt(self._fit("never official, never on a public board" if unlimited
                                else "(Settings > Wing limits: Real / Unlimited)", self.f_lbl, wmax),
                      x, y, self.f_lbl, C_TEXT_DIM)

    def _hint_lines(self, hint: str, avoid=None) -> list:
        """Where the hint goes, [(line, x, y)]: its OWN lines just over the
        bar, never on the bar's key lines (task 45: a short hint used to be
        written over the first of them). Right-aligned to the bar, or left of
        `avoid` (the wing tutorial's box) where that comes down beside them.
        One or two lines ending on the legend's row, right of the vectors'
        legend; a hint too long for two lines that narrow is lifted a row,
        over the legend, to the full width -- and cut with '...' only when
        two lines of that cannot hold it either."""
        u, f = self.ui, self.f_lbl
        lh, low = 18 * u, 668 * u              # the legend's row
        left, right = 22 * u, 1258 * u
        if (avoid is not None and avoid.bottom > low - 2 * lh and avoid.y < low + lh
                and avoid.x > left):
            right = min(right, avoid.x - 10 * u)
        side = right - max(left, self._legend_r + 16 * u)
        lines = self._wrap_px(hint, f, side) if side > 80 * u else []
        if not lines or len(lines) > 2:
            lines = self._wrap_px(hint, f, right - left)
            if self._legend_r:
                low -= lh
            if len(lines) > 2:
                lines = [lines[0], self._fit(" ".join(lines[1:]), f, right - left)]
        n = len(lines)
        return [(s, right - f.size(s)[0], low - (n - 1 - i) * lh) for i, s in enumerate(lines)]

    def _draw_help(self, pad_name, hint, alpha: float = 1.0, avoid=None) -> None:
        u = self.ui
        r = self._panel((12, 690, 1256, 98))
        x, y = r.x + 10 * u, r.y + 6 * u
        #  task 46: L / G are the SAVED WINGS / SAVED CARS card pages, K saves
        #  the selected slot's wing under a name (the pad: OPTIONS' rows)
        lines = [
            ("GARAGE  -  wings: side (both sides, mirrored), top.  1 2 3 select, arrows place, W wing, "
             "D design, A airfoils, L saved wings, G saved cars", C_TEXT),
            ("mouse drag orbit | wheel zoom | LEFT/RIGHT x | UP/DOWN h | [ ] incidence | M mirror | "
             "T top mode | SPACE deploy | V forces | R R reset (U undo) | C camera", C_TEXT_DIM),
            ("ENTER drive  |  S save build  K save wing  B next build  F car default  |  "
             "ESC: change car, wing tutorial, controls, main menu, quit", C_TEXT_DIM),
        ]
        if pad_name:
            lines[2] = (f"PS5 {pad_name}:  L-stick move | R-stick orbit | L1/R1 incidence | "
                        "TRIANGLE slot | SQUARE wing | CIRCLE deploy | L3 design | CROSS drive | "
                        "OPTIONS menu, builds", C_OK)
            lines.append(("ENTER drive  |  S save build  K save wing  B next build  F car default  |  "
                          "ESC: change car, wing tutorial, controls, main menu, quit", C_TEXT_DIM))
        else:
            #  the drive's words (input.MENU_NO_PAD, task 45)
            lines.append(("no gamepad connected - plug one in any time", C_TEXT_DIM))
        for s, c in lines:
            self._txt(s, x, y, self.f_lbl, c)
            y += 18 * u
        #  the hint, however short, on its own lines over the bar (task 45;
        #  task 41 had put only the long ones there), fading as it goes
        self.hint_drawn = self._hint_lines(hint, avoid) if hint and alpha > 0.0 else []
        for s_, hx, hy in self.hint_drawn:
            self._txt(s_, hx, hy, self.f_lbl, C_PANEL_ON, alpha)


# =========================================================================== #
#  THE LIBRARY SINGLETON                                                       #
# =========================================================================== #
_LIB: "Library | None" = None


def library(root: str | None = None) -> Library:
    """The process-wide library (runs/library); XFOIL on when the binary is."""
    global _LIB
    if _LIB is None or (root is not None and _LIB.root != root):
        _LIB = Library(root or os.path.join("runs", "library"))
    return _LIB


def _section_metrics(lib: Library, name: str, re: float, cl_design: float) -> dict:
    pol = lib.polar(name, re, want_xfoil=False)
    g = lib.airfoils[name].geometry()
    cl_d = min(max(cl_design, pol.cl_min + 0.05), pol.cl_max - 0.05)
    cd_d = float(pol.cd_of_cl(cl_d))
    return dict(name=name, cl_max=pol.cl_max, ld_cr=(cl_d / cd_d if cd_d > 1e-6 else 0.0),
                ld_max=pol.ld_max, tc=g["tc"], camber=g["camber"], cm=abs(pol.cm0),
                a_lin=pol.a_lin, alpha_L0=pol.alpha_L0_deg, source=pol.source, polar=pol)


# =========================================================================== #
#  THE DESIGN ENGINE: AeroBO's own (drive/aerobo_models.py)                    #
# =========================================================================== #
#  carsim's wing designer lived here -- its objective, its lattice search, its
#  random-search control, its section model. The owner asked for AeroBO
#  itself inside the game, so the DESIGN page's models are now
#  `aerobo_models` (sessions on the vendored engine) and nothing here
#  designs a wing. The module is imported LAZILY: `import aerobo.api` costs
#  about a second, and the drive's launch reads this module for `CarBuild`
#  alone.
def _am():
    from . import aerobo_models
    return aerobo_models


def wing_name(key: str) -> str:
    """The slot's wing in words: "side wing", "top wing" (not "top wing
    wing")."""
    label = SLOT_LABEL[key].lower()
    return label if label.endswith("wing") else f"{label} wing"


class _SearchView:
    """`Garage.search`, the dict the older callers read ("mode", "effort",
    "stop_early"), as a VIEW of the garage's `aerobo_models.SearchPolicy`:
    one policy, so the two spellings can never disagree."""

    def __init__(self, policy):
        self._p = policy

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        p = self._p
        if key == "mode":
            return p.mode
        if key == "effort":
            return p.effort
        if key in ("stop_early", "stop_when_converged"):
            return bool(p.stop_when_converged)
        raise KeyError(key)

    def __setitem__(self, key, value) -> None:
        p = self._p
        if key == "mode":
            p.set_mode(str(value))
        elif key == "effort":
            p.set_effort(str(value))
        elif key in ("stop_early", "stop_when_converged"):
            p.stop_when_converged = bool(value)
        else:
            raise KeyError(key)

# =========================================================================== #
#  THE AIRFOIL PAGE                                                            #
# =========================================================================== #
class AirfoilPage:
    def __init__(self, g: "Garage"):
        self.g, self.lib = g, g.lib
        self.list = ui.ListBox(title="SECTION LIBRARY  (ranked)")
        self.weights = dict(ld_cr=0.35, cl_max=0.20, ld_max=0.15, thin=0.10, cm=0.20)
        self.sort = "score"
        self.cl_design = 0.8
        self._metrics: dict = {}
        self._re = 1e6
        P = ui.Param
        self.params = ui.ParamList([
            P("w", "RANKING WEIGHTS (WingLab screen)", None, kind="label"),
            P("ld_cr", "L/D at design cl", lambda: self.weights["ld_cr"], self._w("ld_cr"), step=0.05, lo=0, hi=1),
            P("cl_max", "max lift (cl_max)", lambda: self.weights["cl_max"], self._w("cl_max"), step=0.05, lo=0, hi=1),
            P("ld_max", "efficiency (L/D max)", lambda: self.weights["ld_max"], self._w("ld_max"), step=0.05, lo=0, hi=1),
            P("thin", "thickness (thin)", lambda: self.weights["thin"], self._w("thin"), step=0.05, lo=0, hi=1),
            P("cm", "pitching moment (|cm|)", lambda: self.weights["cm"], self._w("cm"), step=0.05, lo=0, hi=1),
            P("cl", "design cl", lambda: self.cl_design, self._set_cl, step=0.1, fine=0.02, lo=0.0, hi=2.0),
            P("sort", "sort by", lambda: self.sort, self._set_sort, kind="choice",
              choices=["score", "cl_max", "ld_cr", "name", "tc"]),
            P("a", "ACTIONS", None, kind="label"),
            #  task 45: dimmed while no wing is being designed -- the page
            #  opened from the car page (A) browses, it has nothing to put
            #  the section on; ENTER on it still says why (`assign_airfoil`)
            P("use", "use this section  (ENTER)", None, lambda _: self.g.assign_airfoil(), kind="action",
              enabled=lambda: self.g._af_for_wing(),
              help="puts the section on the wing being designed; with none open, "
                   "D on the car page designs one"),
            P("xf", "XFOIL polar for it  (X)", None, lambda _: self.request_xfoil(), kind="action",
              enabled=self.lib.use_xfoil),
            P("new", "new NACA 4-digit  (N)", None, lambda _: self.g.prompt_naca(), kind="action"),
        ], title="")
        self.focus = "list"

    def _w(self, key):
        def f(v):
            self.weights[key] = float(v)
            self.refresh()
        return f

    def _set_cl(self, v):
        self.cl_design = float(v)
        self._metrics.clear()
        self.refresh()

    def _set_sort(self, v):
        self.sort = v
        self.refresh()

    def open(self, re: float, cl_design: float | None = None, keep: str | None = None) -> None:
        self._re = re_bank_snap(re)
        if cl_design is not None:
            self.cl_design = float(min(max(cl_design, 0.1), 2.0))
        self._metrics.clear()
        self.refresh(keep=keep)

    def metrics(self, name: str) -> dict:
        m = self._metrics.get(name)
        if m is None:
            try:
                m = _section_metrics(self.lib, name, self._re, self.cl_design)
            except Exception as exc:               # a bad section never kills the page
                m = dict(name=name, cl_max=0.0, ld_cr=0.0, ld_max=0.0, tc=0.0, camber=0.0, cm=1.0,
                         a_lin=0.0, alpha_L0=0.0, source="error", polar=None, error=str(exc))
            self._metrics[name] = m
        return m

    def refresh(self, keep: str | None = None) -> None:
        keep = keep or (self.list.current()[0] if self.list.current() else None)
        rows = [self.metrics(n) for n in sorted(self.lib.airfoils)]
        ok = [m for m in rows if m.get("polar") is not None]

        def norm(key, invert=False):
            vals = np.asarray([m[key] for m in ok], float)
            lo, hi = float(vals.min()), float(vals.max())
            s = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
            return 1.0 - s if invert else s

        if ok:
            score = (self.weights["ld_cr"] * norm("ld_cr") + self.weights["cl_max"] * norm("cl_max")
                     + self.weights["ld_max"] * norm("ld_max") + self.weights["thin"] * norm("tc", True)
                     + self.weights["cm"] * norm("cm", True))
            wsum = sum(self.weights.values()) or 1.0
            for m, sc in zip(ok, score):
                m["score"] = 100.0 * float(sc) / wsum
        for m in rows:
            m.setdefault("score", 0.0)
        keyf = {"score": lambda m: -m["score"], "cl_max": lambda m: -m["cl_max"], "ld_cr": lambda m: -m["ld_cr"],
                "name": lambda m: m["name"], "tc": lambda m: m["tc"]}[self.sort]
        rows.sort(key=keyf)
        items = []
        for m in rows:
            src = {"xfoil": "XFOIL", "estimate": "est", "error": "ERR"}.get(m["source"], "?")
            sub = (f"t/c {100 * m['tc']:4.1f}%  camber {100 * m['camber']:4.1f}%  clmax {m['cl_max']:.2f}  "
                   f"L/D@cl {m['ld_cr']:5.1f}  |cm| {m['cm']:.3f}  [{src}]")
            items.append((m["name"], sub, f"{m['score']:5.1f}"))
        self.list.set_items(items, keep=keep)

    def current_name(self) -> str | None:
        it = self.list.current()
        return it[0] if it else None

    def request_xfoil(self) -> None:
        name = self.current_name()
        if name and self.lib.use_xfoil:
            self.lib.polar(name, self._re, want_xfoil=True)
            self.g.say(f"XFOIL queued for {name}")

    def on_polar(self, name: str) -> None:
        self._metrics.pop(name, None)
        self.refresh()

    def draw(self, screen, text: ui.Text, plot: ui.Plot, u: float) -> str:
        R = lambda x, y, w, h: (int(x * u), int(y * u), int(w * u), int(h * u))   # noqa: E731
        self.list.draw(screen, text, ui.panel(screen, R(12, 12, 528, 664)), row_h=int(33 * u), size=13,
                       focus=(self.focus == "list"))
        help_ = self.params.draw(screen, text, ui.panel(screen, R(548, 12, 300, 300)), row_h=int(21 * u), size=13,
                                 focus=(self.focus == "params"))
        name = self.current_name()
        r_sec = ui.panel(screen, R(856, 12, 412, 300))
        r_pol = ui.panel(screen, R(548, 320, 356, 250))
        r_drg = ui.panel(screen, R(912, 320, 356, 250))
        r_txt = ui.panel(screen, R(548, 578, 720, 98))
        if name:
            m = self.metrics(name)
            p = m.get("polar")
            try:
                coords = self.lib.airfoils[name].coords()
            except Exception:
                coords = None
            if coords is not None:
                plot.begin(screen, r_sec, (-0.05, 1.05), (-0.32, 0.32), title=name, equal=True)
                plot.fill(coords[:, 0], coords[:, 1], (60, 62, 70))
                plot.line(coords[:, 0], coords[:, 1], C_TEXT, 2, closed=True)
                plot.hline(0.0, ui.C_GRID)
            if p is not None:
                plot.begin(screen, r_pol, (p.alpha.min(), p.alpha.max()), (min(p.cl.min(), -0.3) - 0.1, p.cl_max + 0.3),
                           title=f"cl(alpha)  {'XFOIL' if p.source == 'xfoil' else 'ESTIMATE'}  Re {p.re:.2g}",
                           xlabel="alpha deg")
                plot.line(p.alpha, p.cl, C_PANEL_ON, 2)
                plot.vline(p.alpha_valid[1], C_WARN)
                plot.hline(self.cl_design, ui.C_LINE4)
                i0 = int(np.argmin(np.abs(p.alpha - p.alpha_valid[0])))
                i1 = int(np.argmin(np.abs(p.alpha - p.alpha_valid[1])))
                cd_b, cl_b = p.cd[i0:i1 + 1], p.cl[i0:i1 + 1]
                plot.begin(screen, r_drg, (0.0, max(float(cd_b.max()) * 1.1, 0.02)), (min(float(cl_b.min()), -0.3), p.cl_max + 0.2),
                           title="drag polar cl(cd)", xlabel="cd")
                plot.line(cd_b, cl_b, ui.C_LINE2, 2)
                plot.hline(self.cl_design, ui.C_LINE4)
                text.blit(screen, f"{name}: {self.lib.airfoils[name].notes}"[:110], r_txt.x + 10, r_txt.y + 8, 12, C_TEXT)
                text.blit(screen, f"score {m['score']:.1f}   a {m['a_lin']:.2f}/rad  alpha_L0 {m['alpha_L0']:+.2f} deg  "
                          f"cl_max {m['cl_max']:.2f}  cd_min {1e4 * p.cd_min:.0f} ct  L/D at cl {self.cl_design:.2f}: {m['ld_cr']:.1f}",
                          r_txt.x + 10, r_txt.y + 30, 12, C_TEXT_DIM)
                text.blit(screen, ("XFOIL polar in the cache" if p.source == "xfoil" else
                          "ESTIMATE polar (panel method + friction correlation): X runs XFOIL in the background"),
                          r_txt.x + 10, r_txt.y + 52, 12, C_OK if p.source == "xfoil" else C_WARN)
            elif m.get("error"):
                text.blit(screen, f"cannot analyse: {m['error']}"[:110], r_txt.x + 10, r_txt.y + 8, 12, C_WARN)
        return help_


# =========================================================================== #
#  THE LIBRARY PAGE                                                            #
# =========================================================================== #
class LibraryPage:
    def __init__(self, g: "Garage"):
        self.g, self.lib = g, g.lib
        self.wings = ui.ListBox(title="WINGS  (ENTER: put in the selected slot)")
        self.builds = ui.ListBox(title="BUILDS  (ENTER: load the whole car)")
        self.other: set = set()          # other cars' builds: drawn dim (task 41)
        self.focus = "wings"
        self.refresh()

    def refresh(self, keep: str | None = None) -> None:
        """Re-list both. `keep`: the build the cursor goes to (a rename,
        task 41); else each list keeps the row it was on."""
        keep_w = self.wings.current()[0] if self.wings.current() else None
        keep_b = self.builds.current()[0] if self.builds.current() else None
        items = []
        for name in sorted(self.lib.wings):
            w = self.lib.wings[name]
            a = w.aero
            if w.legacy:
                sub = f"published panel: CL0 {w.legacy['CL0']:.2f}  L/D {w.legacy['LD']:.1f}  S {w.legacy.get('S', 0.35):.2f}"
            elif "CLa" in a:
                src = ("WingLab" if w.engine == "aerobo" else
                       "est" if a.get("polar_is_estimate", True) else "XFOIL")
                sub = (f"{w.airfoil}  b {w.span:.2f} c {w.chord:.2f} taper {w.taper:.2f} plates {w.plate_h:.2f}  "
                       f"S {w.S:.2f}  CLa {a['CLa']:.2f}  CLmax {a['CL_max']:.2f}  {src}")
            else:
                sub = f"{w.airfoil}  b {w.span:.2f} c {w.chord:.2f}  (not analysed)"
            #  task 45: a built-in reads by its player name, with what it
            #  does ahead of its numbers, and says 'built in' (it was a bare
            #  '*'); the row's key -- what ENTER, R and DEL act on -- is still
            #  the library name, the name shown is the row's 4th field
            shown, what = wing_shown(name, self.lib)
            items.append((name, f"{what}  |  {sub}" if what else sub,
                          f"{w.role}{', built in' if w.builtin else ''}", shown))
        self.wings.set_items(items, keep=keep_w)
        #  task 41: the BUILDS of the car in the garage first, then the any-car
        #  ones saved before builds knew their car, then every other car's --
        #  the PICK page's order (`prerace.pick_order`), which B cycles too.
        #  Another car's build stays loadable (a bus wing on a Corsa is the
        #  player's own experiment) but is tagged with its car and drawn dim;
        #  the car's own default is tagged "(default)".
        from .prerace import pick_order, car_label
        car, dflt = self.g.car, self.g.default_name()
        self.builds.title = f"BUILDS for the {car_label(car)}  (ENTER: load the whole car)"
        items, self.other = [], set()
        for name in pick_order(self.lib.builds, car):
            b = CarBuild.from_json(self.lib.builds[name])
            tag = car_label(b.car) if b.car else "any car"
            sub = b.summary(self.lib)[:90]
            if b.car and b.car != car:
                self.other.add(name)
                sub = f"the {car_label(b.car)}'s: " + sub
            items.append((name, sub, ("(default) " if name == dflt else "") + tag))
        self.builds.set_items(items, keep=keep_b if keep is None else keep)

    def draw(self, screen, text: ui.Text, u: float) -> None:
        R = lambda x, y, w, h: (int(x * u), int(y * u), int(w * u), int(h * u))   # noqa: E731
        self.wings.draw(screen, text, ui.panel(screen, R(12, 12, 640, 336)), row_h=int(33 * u), size=13,
                        focus=(self.focus == "wings"))
        self.builds.draw(screen, text, ui.panel(screen, R(12, 356, 640, 320)), row_h=int(33 * u), size=13,
                         focus=(self.focus == "builds"), empty="(no saved builds: S saves the current car)")
        #  another car's build, dimmed: a veil of the panel colour over its row
        #  (the list draws every row alike; the rows it drew are its `_hits`)
        for i, ry, rh in getattr(self.builds, "_hits", ()):
            if (self.builds.items[i][0] in self.other
                    and not (i == self.builds.idx and self.focus == "builds")):
                veil = pygame.Surface((self.builds._rect.w - 8, rh - 2), pygame.SRCALPHA)
                veil.fill((*C_HUD_BG, 120))
                screen.blit(veil, (self.builds._rect.x + 4, ry - 2))
        r = ui.panel(screen, R(660, 12, 608, 664))
        x, y = r.x + 12, r.y + 10
        b = self.g.build
        text.blit(screen, f"CURRENT CAR  '{b.name}'   selected slot: {SLOT_LABEL[self.g.sel]}", x, y, 14, ui.C_SECTION, bold=True)
        y += 24
        for key in SLOTS:
            s = b.slot(key)
            w = self.lib.wings.get(s.wing)
            text.blit(screen, f"{SLOT_LABEL[key]:12s} {wing_shown(w.name, self.lib)[0] if w else 'none':18s} "
                      f"x {s.x:+.2f}  h {s.h:.2f}  inc {s.inc_deg:+.1f}"
                      + (f"  {s.mode}" if key == "top" else ""), x, y, 13, C_TEXT if w else C_TEXT_DIM)
            y += 20
        y += 10
        it = self.wings.current() if self.focus == "wings" else self.builds.current()
        if it and self.focus == "wings":
            w = self.lib.wings[it[0]]
            shown, what = wing_shown(w.name, self.lib)
            text.blit(screen, f"WING '{shown}'  ({w.role}{', built in' if w.builtin else ''})", x, y, 14, ui.C_SECTION, bold=True)
            y += 22
            #  task 45: the built-ins' notes are saved in the library as the
            #  study wrote them ("the study's clean fin: ..."); shown plainly,
            #  after what the wing does (a built-in's `wing_shown` line)
            notes = (w.notes or "").replace("the study's ", "")
            lines = [what, notes, f"section {w.airfoil}   span {w.span:.3f} m   chord {w.chord:.3f} m   taper {w.taper:.2f}",
                     f"twist {w.twist_deg:+.1f} deg   plates {w.plate_h:.3f} m   S {w.S:.3f} m2   AR {w.AR:.2f}   MAC {w.mac:.3f} m"]
            a = w.aero
            if "CLa" in a:
                lines.append(f"CL = {a['CL0']:+.3f} + {a['CLa']:.3f} alpha  in [{a['CL_min']:+.2f}, {a['CL_max']:+.2f}]   e {a['e']:.3f}   "
                             + ("law sampled from WingLab's evaluator" if w.engine == "aerobo" else
                                f"polar {'ESTIMATE' if a.get('polar_is_estimate', True) else 'XFOIL'}")
                             + f" Re {a.get('Re', 0):.2g}")
                lines.append(f"CD = {a['cd0']:.4f} {a['cd1']:+.4f} CL {a['cd2']:+.4f} CL^2")
                dp = design_point(w, 0.0)
                if dp:
                    lines.append(f"at {dp['V']:.1f} m/s, 0 deg:  CL {dp['CL']:.2f}  F {dp['F']:.0f} N  D {dp['D']:.1f} N  L/D {dp['LD']:.1f}")
            elif w.legacy:
                lines.append(f"closed form: CL0 {w.legacy['CL0']:.2f}, L/D {w.legacy['LD']:.1f}, S {w.legacy.get('S', 0.35):.2f} m2")
            for ln in lines:
                if ln:
                    text.blit(screen, ln[:90], x, y, 12, C_TEXT_DIM)
                    y += 18
            role_ok = w.role == SLOT_ROLE[self.g.sel]
            y += 8
            text.blit(screen, ("ENTER puts it in the selected slot" if role_ok else
                      f"a {w.role} wing does not fit the {self.g.sel} slot: select a {'flank' if w.role == 'flank' else 'top'} slot first (1/2/3)"),
                      x, y, 12, C_OK if role_ok else C_WARN)
        elif it and self.focus == "builds":
            from .prerace import car_label
            bb = CarBuild.from_json(self.lib.builds[it[0]])
            car = self.g.car
            text.blit(screen, f"BUILD '{bb.name}'" + ("   (default)" if bb.name == self.g.default_name()
                                                     else ""), x, y, 14, ui.C_SECTION, bold=True)
            y += 22
            for key in SLOTS:
                s = bb.slot(key)
                text.blit(screen, f"{SLOT_LABEL[key]:12s} {wing_shown(s.wing, self.lib)[0] if s.wing else 'none':18s} "
                          f"x {s.x:+.2f}  h {s.h:.2f}  inc {s.inc_deg:+.1f}",
                          x, y, 13, C_TEXT_DIM)
                y += 20
            y += 8
            #  whose it is (task 41), and whether its wings are past THIS
            #  car's span limits (`bodies.over_limits`: computed, never stored)
            if not bb.car:
                whose, col = "for any car (saved before builds knew their car)", C_TEXT_DIM
            elif bb.car == car:
                whose, col = f"made for this car, the {car_label(car)}", C_OK
            else:
                whose, col = (f"made for the {car_label(bb.car)}: on the {car_label(car)} its "
                              f"slots are fitted to this body"), C_WARN
            text.blit(screen, whose[:90], x, y, 12, col)
            y += 18
            try:
                from . import bodies
                over = bodies.over_limits(self.lib.builds[it[0]], self.lib, car)
            except Exception:              # noqa: BLE001 -- a tag is never worth a crash
                over = []
            if over:
                text.blit(screen, f"UNLIMITED on the {car_label(car)}: {bodies.limits_text(over)}"[:90],
                          x, y, 12, C_WARN)
                y += 18
            y += 6
            text.blit(screen, "ENTER loads it as the current car   D makes it the car's default   "
                      "R renames it", x, y, 12, C_OK)
        y = r.bottom - 60
        text.blit(screen, "TAB wings/builds   ENTER use/load   S save car (SHIFT: as new)   "
                  "N new wing", x, y, 12, C_TEXT_DIM)
        text.blit(screen, "D the car's default build   R rename   DEL twice: delete (user items)   "
                  "ESC back", x, y + 16, 12, C_TEXT_DIM)
        if self.g.hint and self.g.hint_alpha() > 0.0:        # gone after HINT_S (task 45)
            text.blit(screen, self.g.hint[:90], x, y + 34, 12, ui.C_KEY)


# =========================================================================== #
#  SAVED WINGS AND SAVED CARS: the two card pages (task 46)                    #
# =========================================================================== #
#  The owner (2026-09-28): "There should be a way to save the wings (same as
#  saving cars) to test them on different cars. (Can only be fitted in cars
#  they fit). Make it easy to see the list of wings (a small diagram)" and
#  "I want there in the garage or similar to see the different saved cars."
#
#  SAVED WINGS (L on the car page): every wing in the library as a card --
#  the player's own first, then the built-ins -- with a small diagram drawn
#  from its WingSpec (the planform and a front view, in the wing colour), its
#  name, role, span, area and the car it was made on, and whether it FITS the
#  selected slot of the car in the garage (`wing_fit`: the slot's role, and
#  this car's span limit at the slot, the rule W holds). One that does not is
#  drawn dimmed with the reason in red, and ENTER on it says what would make
#  it fit. SAVED CARS (G): every saved build as a card with a small picture
#  of its car carrying its wings -- the car page's own GarageView drawing into
#  an offscreen surface, cached per build. The text library (`LibraryPage`,
#  SHIFT+L) stays for its numbers. K on the car page saves the selected
#  slot's wing under a name the player types (`Garage.prompt_wing`).

#: spans up to this many metres share ONE scale on the wing cards, so a
#: 0.63 m and a 1.88 m side wing look their sizes (and chords up to ~0.85 m:
#: the plan view is 56 % of the diagram's height); a longer or deeper wing
#: (a bus's) is fitted to its card. Each plan view carries a scale bar.
DIAG_M_MAX = 2.2
#: the plan view's share of the diagram's height (the front view the rest)
DIAG_PLAN_SHARE = 0.56
#: the section thickness the front view gives a wing, over its chord (a
#: diagram: the car page lofts the real section)
DIAG_TC = 0.12
#: the pylons' colour in a diagram (C_STRUT is drawn on the lit car; on the
#: dark card it would all but vanish)
C_DIAG_STRUT = (150, 154, 162)
#: the SAVED CARS pictures drawn in one frame at most (each 5-15 ms): the
#: page opens at once and its cards fill in over the next few frames
PICTURES_PER_FRAME = 2
#: the pictures' camera: its distance over the car page's own, its pitch,
#: and how much higher than the page's it looks (so a top wing is in the
#: frame): the car and its wings fill the card, wheels to wing tips
PICTURE_ZOOM = 0.68
PICTURE_PITCH_DEG = 15.0
PICTURE_LIFT = 1.25


def car_long(car) -> str:
    """A car as a reason names it (task 46): the first two words of its title
    -- 'Opel Corsa', 'Renault Express', 'Mercedes Citaro' -- for a key
    `cars.py` carries, else the key itself."""
    import cars as _cars
    t = _cars.CAR_TITLES.get(car or "corsa")
    return " ".join(str(t).split()[:2]) if t else str(car or "car")


def _known_car(car) -> bool:
    """Is `car` a car the player can pick now (`cars.CAR_ORDER`, the menu's
    Change car list)? A build tagged with a car taken out of the game is
    offered as an any-car build (loaded on this car), never a change of car
    the drive would refuse."""
    import cars as _cars
    return bool(car) and car in _cars.CAR_ORDER and car in _cars.CARS


def wing_fit(spec: "WingSpec", key: str, build: "CarBuild", car=None,
             unlimited: bool = False) -> tuple:
    """Does library wing `spec` fit slot `key` of `build` on `car`? Task 46:
    (fits, why, limit, past).

      * its ROLE must be the slot's: a top wing goes in the top slot only;
      * its SPAN must be within THIS car's span limit at the slot -- a side
        wing's lower tip at the car's ground clearance for the slot's
        height, a top wing 1.2 x the body's width (`bodies.span_limit`,
        with `over_limits`' tolerance) -- the rule W and the library page
        hold in Real mode; under Settings > Wing limits: Unlimited within
        `bodies.UNLIMITED_FACTOR` (3x) of it, the designer's span ceiling.

    `why` is the refusal a player reads ('too wide for the Opel Corsa:
    1.88 m, the limit here is 1.50 m', 'a top wing: select the top slot'),
    "" when it fits; `limit` the slot's REAL limit (None on a role
    mismatch); `past` a wing that fits only because of Unlimited, whose runs
    then count as UNLIMITED."""
    role = SLOT_ROLE[key]
    if spec.role != role:
        return (False, ("a top wing: select the top slot" if spec.role == "top"
                        else "a side wing: select a side slot"), None, False)
    lim = bodies.span_limit(role, car, build.slot(key).h)
    cap = lim * (bodies.UNLIMITED_FACTOR if unlimited else 1.0)
    span = float(spec.span)
    if span > cap + bodies.LIMIT_TOL:
        return (False, (f"too wide for the {car_long(car)}"
                        + (" even in Unlimited" if unlimited else "")
                        + f": {span:.2f} m, the limit here is {cap:.2f} m"), lim, False)
    return True, "", lim, span > lim + bodies.LIMIT_TOL


def fit_height(spec: "WingSpec", car=None, unlimited: bool = False) -> float | None:
    """The lowest side-slot height at which side wing `spec` is within `car`'s
    span limit (a side wing's limit grows with its height: 2 (h - ground),
    x3 in Unlimited), when a slot of the car can go that high; else None --
    and None for a top wing, whose limit is the car's width wherever it is
    mounted. What ENTER on a wing too wide for the slot suggests (task 46)."""
    if spec.role != "flank":
        return None
    k = bodies.UNLIMITED_FACTOR if unlimited else 1.0
    h = bodies.body(car).ground + 0.5 * float(spec.span) / k
    return h if h <= bodies.flank_h_band(car)[1] + 1e-9 else None


def wing_origin(lib: "Library", name: str) -> tuple[str, str]:
    """Where library wing `name` comes from, for its card (task 46): (a car
    key, how). 'made' -- the wing records the car it was made on
    (`WingSpec.made_for`); 'used' -- it records none (saved before task 46)
    but the saved builds carrying it are all one car's, so it is shown as
    that car's (the owner's 1.88 m WingLab side wing rides in 'my express
    (autosave)'); ('', 'built') a built-in, every car's; ('', '') nothing
    says."""
    w = lib.wings.get(name)
    if w is None:
        return "", ""
    if w.builtin:
        return "", "built"
    if w.made_for:
        return w.made_for, "made"
    cars_ = set()
    for js in lib.builds.values():
        if not isinstance(js, dict):
            continue
        sl = js.get("slots") if isinstance(js.get("slots"), dict) else {}
        if any(isinstance(v, dict) and v.get("wing") == name for v in sl.values()):
            c = js.get("car", "")
            cars_.add(c if isinstance(c, str) else "")
    cars_.discard("")
    return (next(iter(cars_)), "used") if len(cars_) == 1 else ("", "")


def origin_text(lib: "Library", name: str) -> str:
    """`wing_origin` in a player's words: 'made on the Express', 'used on
    the Express', 'built in: on every car', 'car not recorded'."""
    from .prerace import car_label
    car, how = wing_origin(lib, name)
    if how == "built":
        return "built in: on every car"
    if how == "made":
        return f"made on the {car_label(car)}"
    if how == "used":
        return f"used on the {car_label(car)}"
    return "car not recorded"


def wing_order(lib: "Library") -> list:
    """The SAVED WINGS page's order (task 46): the player's own wings first,
    then the built-ins, each group by name."""
    ws = lib.wings
    return sorted(ws, key=lambda n: (bool(ws[n].builtin), n.lower(), n))


def _shade(col, k: float) -> tuple:
    """`col` taken toward the panel colour: `k` 1 is itself, 0 the panel."""
    return tuple(int(round(C_HUD_BG[i] + (col[i] - C_HUD_BG[i]) * k)) for i in range(3))


def wing_diagram(screen, rect, spec: "WingSpec", text: "ui.Text", u: float = 1.0,
                 dim: bool = False) -> dict:
    """The small diagram of `spec` in `rect` (task 46): two views stacked.

    PLAN   the planform, span across and chord down (the leading edge up),
           drawn round the mid-chord line as the car page lofts it: the root
           chord at the centre, `taper` of it at the tips; the tip devices
           as bars across the tips (their chord); a 1 m bar top right.
    FRONT  the wing seen end on, span across: its section (DIAG_TC of the
           local chord) on top, the tip devices / endplates at their height
           (`plate_h_flown`) leaning by `plate_cant_deg`, and the MOUNT down
           to the body -- the car's side for a side wing, the deck for a top
           wing: two pylons at +-`pylon_frac` of the semi-span, or the
           endplates themselves running to it (a WingLab plate that stops
           short of the body bracketed on, as the car page draws it).

    ONE scale for both views: DIAG_M_MAX metres across the box, a longer or
    deeper wing fitted to it; the gap down to the body is clipped to the box
    (a diagram, not a drawing to scale). In the wing colour -- C_PANEL_ON,
    the plates C_PLATE -- shaded toward the panel when `dim` (a wing that
    does not fit). Returns what it drew: px per metre, the two views' rects,
    the span and body line in px, and the plates and pylons drawn."""
    from .aero.wing import TOP_PYLON_L
    r = pygame.Rect(rect)
    k = 0.40 if dim else 1.0
    c_wing, c_plate = _shade(C_PANEL_ON, k), _shade(C_PLATE, k)
    c_fill = _shade(C_PANEL_ON, 0.45 * k)
    c_strut, c_body = _shade(C_DIAG_STRUT, k), _shade(C_TEXT_DIM, 0.55 * k + 0.1)
    c_lab = _shade(C_TEXT_DIM, 0.8 * k + 0.2)
    lab, pad = int(12 * u), int(6 * u)
    hp = int((r.h - int(4 * u)) * DIAG_PLAN_SHARE)
    plan = pygame.Rect(r.x, r.y, r.w, hp)
    front = pygame.Rect(r.x, plan.bottom + int(4 * u), r.w, r.bottom - plan.bottom - int(4 * u))
    for box in (plan, front):
        pygame.draw.rect(screen, _shade(C_BG, 1.0), box)
    wmax = r.w - 2 * pad
    b, c = max(float(spec.span), 1e-3), max(float(spec.chord), 1e-3)
    lam = min(max(float(spec.taper), 0.0), 2.0)
    s = min(wmax / DIAG_M_MAX, wmax / b, (plan.h - lab - pad) / (c * max(1.0, lam)))
    cx, hb = r.centerx, 0.5 * b * s
    out = dict(px_per_m=s, plan=plan, front=front, span_px=2.0 * hb, plates=0, pylons=0)
    # --- plan ---------------------------------------------------------------
    text.blit(screen, "plan", plan.x + pad, plan.y + 2, 10, c_lab)
    #  the scale bar: half a metre, as long as that is legible (a bus wing's
    #  card is drawn smaller: a metre, or two)
    bar_m = next((m_ for m_ in (0.5, 1.0, 2.0) if m_ * s >= 16 * u), 5.0)
    bar = bar_m * s
    by, bx1 = plan.y + int(7 * u), plan.right - pad
    pygame.draw.line(screen, c_lab, (int(bx1 - bar), by), (bx1, by), 1)
    for xx in (bx1 - bar, bx1):
        pygame.draw.line(screen, c_lab, (int(xx), by - 3), (int(xx), by + 3), 1)
    text.blit(screen, f"{bar_m:g} m", bx1 - bar - 5, plan.y + 2, 10, c_lab, right=True)
    out["bar_m"] = bar_m
    cy = plan.y + lab + (plan.h - lab) // 2
    cr, ct = 0.5 * c * s, 0.5 * c * lam * s
    poly = [(cx - hb, cy - ct), (cx, cy - cr), (cx + hb, cy - ct),
            (cx + hb, cy + ct), (cx, cy + cr), (cx - hb, cy + ct)]
    pygame.draw.polygon(screen, c_fill, poly)
    pygame.draw.polygon(screen, c_wing, poly, max(1, int(round(1.5 * u))))
    ph = float(spec.plate_h_flown)
    if ph > 0.0:
        ratio = float(spec.plate_chord_ratio) or 1.3
        pc = 0.5 * c * lam * ratio * s
        for sg in (-1.0, 1.0):
            x = cx + sg * hb
            pygame.draw.line(screen, c_plate, (int(x), int(cy - pc)), (int(x), int(cy + pc)),
                             max(2, int(round(3 * u))))
    # --- front ----------------------------------------------------------------
    flank = spec.role == "flank"
    text.blit(screen, "front, car side below" if flank else "front, deck below",
              front.x + pad, front.y + 2, 10, c_lab)
    cant = math.radians(min(max(float(spec.plate_cant_deg), 5.0), 90.0))
    carried = spec.mount == "endplate"
    if flank:
        gap_m = flank_out(spec, 1.0)
    else:
        gap_m = max(ph * math.sin(cant), 0.10) if carried else TOP_PYLON_L
    t_r = max(2.0, DIAG_TC * c * s)
    t_t = max(2.0, DIAG_TC * c * lam * s)
    wy = front.y + lab + int(3 * u) + 0.5 * t_r
    body_y = min(wy + gap_m * s, front.bottom - int(6 * u))
    out["body_y"] = body_y
    x0, x1 = front.x + pad, front.right - pad
    pygame.draw.line(screen, c_body, (x0, int(body_y)), (x1, int(body_y)), 1)
    step = max(5, int(7 * u))
    for hx in range(x0 + step, x1, step):
        pygame.draw.line(screen, c_body, (hx, int(body_y) + 1), (hx - int(4 * u), int(body_y) + int(4 * u)), 1)
    if spec.mount == "pylon":
        for sg in (-1.0, 1.0):
            xp = int(cx + sg * float(spec.pylon_frac) * hb)
            pygame.draw.line(screen, c_strut, (xp, int(wy)), (xp, int(body_y)), max(2, int(round(2 * u))))
            out["pylons"] += 1
    reach = max(body_y - wy, 0.0)
    to_body = reach / max(math.sin(cant), 0.2)
    L = to_body if (carried and spec.engine != "aerobo") else min(ph * s, to_body)
    if ph > 0.0 or carried:
        for sg in (-1.0, 1.0):
            xa = cx + sg * hb
            xb, yb = xa + sg * math.cos(cant) * L, wy + math.sin(cant) * L
            if L >= 1.0:
                pygame.draw.line(screen, c_plate, (int(xa), int(wy)), (int(xb), int(yb)),
                                 max(2, int(round(3 * u))))
                out["plates"] += 1
            if carried and yb < body_y - 1.0:         # a WingLab plate short of the body
                pygame.draw.line(screen, c_strut, (int(xb), int(yb)), (int(xb), int(body_y)), 1)
    wpoly = [(cx - hb, wy - 0.5 * t_t), (cx, wy - 0.5 * t_r), (cx + hb, wy - 0.5 * t_t),
             (cx + hb, wy + 0.5 * t_t), (cx, wy + 0.5 * t_r), (cx - hb, wy + 0.5 * t_t)]
    pygame.draw.polygon(screen, c_wing, wpoly)
    return out


def _wrap_lines(text: "ui.Text", s: str, px: float, size: int, n_max: int) -> list:
    """`s` broken at its spaces into the lines that fit `px` at `size`, at
    most `n_max` of them (the last cut with '...' when it has to be)."""
    lines = ui._wrap_px(text, s, int(px), size) or [""]
    #  a unit never stands alone: '1.05' / 'm' is '1.05 m' on the next line
    for i in range(1, len(lines)):
        prev = lines[i - 1].split(" ")
        if len(lines[i]) <= 3 and " " not in lines[i] and len(prev) > 1:
            lines[i - 1], lines[i] = " ".join(prev[:-1]), f"{prev[-1]} {lines[i]}"
    if len(lines) > n_max:
        lines = lines[:n_max - 1] + [" ".join(lines[n_max - 1:])]
    out = []
    for ln in lines:
        if text.width(ln, size) > px:
            while ln and text.width(ln + "...", size) > px:
                ln = ln[:-1]
            ln = ln.rstrip() + "..."
        out.append(ln)
    return out


def _cut(text: "ui.Text", s: str, px: float, size: int, bold: bool = False) -> str:
    """`s` cut to `px` at `size`, ending '...' when it had to be."""
    if text.width(s, size, bold) <= px:
        return s
    while s and text.width(s + "...", size, bold) > px:
        s = s[:-1]
    return s.rstrip() + "..."


class _CardGrid:
    """A page of cards in rows of `cols`, with a cursor (task 46): the SAVED
    WINGS and SAVED CARS pages. LEFT / RIGHT step a card, UP / DOWN a row
    (`nav`, `nav_row`); the view scrolls by whole rows to keep the cursor in
    sight. The mouse as a list's: a click selects the card under it, a click
    on the selected card is its ENTER (`click` -> 'activate'), the wheel
    steps a row. `names` are the cards' keys (library names)."""

    def __init__(self, cols: int = 3):
        self.cols = cols
        self.names: list = []
        self.idx = 0
        self.top = 0                   # the first row in view
        self._rect = None              # the cards' area, as last drawn
        self._hits: list = []          # [(index, rect)] as last drawn
        self.rows_vis = 1

    def set_names(self, names, keep=None) -> None:
        """The cards, the cursor kept on `keep` (default the card it is on)
        when that is still there, else on the same place in the list."""
        was = self.current() if keep is None else keep
        self.names = list(names)
        if was in self.names:
            self.idx = self.names.index(was)
        self.idx = max(0, min(self.idx, len(self.names) - 1)) if self.names else 0

    def current(self):
        if not self.names:
            return None
        self.idx = max(0, min(self.idx, len(self.names) - 1))
        return self.names[self.idx]

    def nav(self, d: int) -> None:
        if self.names:
            self.idx = (self.idx + d) % len(self.names)

    def nav_row(self, d: int) -> None:
        """A row up / down; off the last row's end, onto its last card."""
        if not self.names:
            return
        j = self.idx + d * self.cols
        if 0 <= j < len(self.names):
            self.idx = j
        elif d > 0 and self.idx // self.cols < (len(self.names) - 1) // self.cols:
            self.idx = len(self.names) - 1

    def hit(self, pos) -> int:
        if pos is None:
            return -1
        for i, rr in self._hits:
            if rr.collidepoint(pos):
                return i
        return -1

    def click(self, pos) -> str:
        i = self.hit(pos)
        if i < 0:
            return ""
        if i == self.idx:
            return "activate"
        self.idx = i
        return "select"

    def wheel(self, dy: int) -> None:
        self.nav_row(-1 if dy > 0 else 1)

    def layout(self, area, card_h: int, gap: int) -> list:
        """[(index, rect)] of the cards in view in `area`, the view scrolled
        to the cursor's row. Kept for `hit` and the self-check."""
        area = pygame.Rect(area)
        self.rows_vis = max(1, (area.h + gap) // (card_h + gap))
        n_rows = (len(self.names) + self.cols - 1) // self.cols
        row = self.idx // self.cols
        if row < self.top:
            self.top = row
        elif row >= self.top + self.rows_vis:
            self.top = row - self.rows_vis + 1
        self.top = max(0, min(self.top, max(0, n_rows - self.rows_vis)))
        cw = (area.w - gap * (self.cols - 1)) // self.cols
        out = []
        for i in range(self.top * self.cols,
                       min(len(self.names), (self.top + self.rows_vis) * self.cols)):
            rr, cc = divmod(i, self.cols)
            out.append((i, pygame.Rect(area.x + cc * (cw + gap),
                                       area.y + (rr - self.top) * (card_h + gap), cw, card_h)))
        self._rect, self._hits = area, out
        return out

    def _card_bg(self, screen, r, sel: bool, hover: bool) -> None:
        """A card's panel: the selected one lit, with the accent border."""
        ui.panel(screen, r, alpha=235 if sel else 200, border=True)
        if sel:
            pygame.draw.rect(screen, C_SEL_BG_CARD, r.inflate(-2, -2))
            pygame.draw.rect(screen, ui.C_ACCENT, r, 2)
        elif hover:
            pygame.draw.rect(screen, ui.C_GRID, r.inflate(-2, -2))

    def more_text(self) -> str:
        """Where the view is, when the cards do not all fit: 'cards 4-15 of
        17 ^v' -- ^ / v when there are more above / below -- ("" when every
        card is in view). The page's key bar shows it at its right, in the
        240 px its title leaves for a status (the cards fill their area to
        the pixel: a mark over it would cover a card)."""
        n = len(self.names)
        first = self.top * self.cols
        last = min(n, (self.top + self.rows_vis) * self.cols)
        if first == 0 and last >= n:
            return ""
        return (f"cards {first + 1}-{last} of {n} " + ("^" if first else "")
                + ("v" if last < n else "") + "  (wheel)")


#: a selected card's fill (a shade over the list's selection colour: the
#: diagram's own dark boxes stand out on it)
C_SEL_BG_CARD = (36, 40, 48)


class SavedWingsPage(_CardGrid):
    """SAVED WINGS (task 46): every library wing as a card with its diagram
    (`wing_diagram`), for the selected slot of the car in the garage. The
    header says which slot and car the cards are judged for, with the slot
    chips a click (or 1 / 2 / 3) changes. `drawn` keeps what the last frame
    drew: [(name, fits, why, rect)] -- the self-check reads it."""

    CARD_H = 148
    DIAG_W = 148

    def __init__(self, g: "Garage"):
        super().__init__(3)
        self.g, self.lib = g, g.lib
        self.drawn: list = []
        self.chips: list = []          # [(slot key, rect)] the header's slot chips
        self.refresh()

    def refresh(self, keep=None) -> None:
        self.set_names(wing_order(self.lib), keep)

    def fit(self, name: str) -> tuple:
        """`wing_fit` of wing `name` for the garage's selected slot and car."""
        g = self.g
        return wing_fit(self.lib.wings[name], g.sel, g.build, g.car, g.unlimited)

    def chip_hit(self, pos) -> str:
        for key, rr in self.chips:
            if pos is not None and rr.collidepoint(pos):
                return key
        return ""

    def draw(self, screen, text: ui.Text, u: float) -> None:
        R = lambda x, y, w, h: pygame.Rect(int(x * u), int(y * u), int(w * u), int(h * u))  # noqa: E731
        g = self.g
        self.set_names(wing_order(self.lib))      # a WingLab commit, a delete: the library now
        #  the header: what the cards are judged for, and the slot chips
        hr = ui.panel(screen, R(12, 12, 1256, 40))
        mine = sum(1 for n in self.names if not self.lib.wings[n].builtin)
        x = hr.x + int(12 * u)
        x += text.blit(screen, "SAVED WINGS", x, hr.y + int(11 * u), 16, ui.C_SECTION, bold=True)
        x += text.blit(screen, f"   {mine} yours, {len(self.names) - mine} built in", x,
                       hr.y + int(13 * u), 12, C_TEXT_DIM) + int(24 * u)
        slot = g.build.slot(g.sel)
        role = SLOT_ROLE[g.sel]
        lim = bodies.span_limit(role, g.car, slot.h)
        where = (f"for the {car_long(g.car)}'s " + ("top slot" if role == "top" else
                                                    f"side slot at h {slot.h:.2f} m")
                 + f": span limit {lim:.2f} m"
                 + (f" (x{bodies.UNLIMITED_FACTOR:.0f} Unlimited)" if g.unlimited else ""))
        self.chips = []
        cx = hr.right - int(12 * u)
        for key in reversed(SLOTS):
            lbl = f"{SLOTS.index(key) + 1} {key}"
            w_ = text.width(lbl, 12) + int(16 * u)
            rr = pygame.Rect(cx - w_, hr.y + int(8 * u), w_, int(24 * u))
            on = key == g.sel
            pygame.draw.rect(screen, C_SEL_BG_CARD if on else C_HUD_BG, rr)
            pygame.draw.rect(screen, ui.C_ACCENT if on else ui.C_BORDER, rr, 2 if on else 1)
            text.blit(screen, lbl, rr.centerx, rr.y + int(5 * u), 12,
                      C_TEXT if on else C_TEXT_DIM, centre=True)
            self.chips.insert(0, (key, rr))
            cx = rr.x - int(6 * u)
        cx -= text.blit(screen, "slot", cx - int(4 * u), hr.y + int(13 * u), 12, C_TEXT_DIM,
                        right=True) + int(20 * u)
        where = _cut(text, where, cx - x, 12)
        text.blit(screen, where, x, hr.y + int(13 * u), 12, C_TEXT)
        #  the cards
        area = R(12, 60, 1256, 622)
        cards = self.layout(area, int(self.CARD_H * u), int(8 * u))
        hover = self.hit(pygame.mouse.get_pos())
        self.drawn = []
        if not self.names:
            text.blit(screen, "(no wings: D designs one)", area.x + 12, area.y + 12, 14, C_TEXT_DIM)
        for i, rr in cards:
            self._card(screen, text, u, self.names[i], rr, i == self.idx, i == hover)

    def _card(self, screen, text, u, name, r, sel, hover) -> None:
        g, w = self.g, self.lib.wings[name]
        fits, why, lim, past = self.fit(name)
        self._card_bg(screen, r, sel, hover)
        dg = pygame.Rect(r.x + int(8 * u), r.y + int(8 * u), int(self.DIAG_W * u),
                         r.h - int(16 * u))
        wing_diagram(screen, dg, w, text, u, dim=not fits)
        x = dg.right + int(10 * u)
        wmax = r.right - int(10 * u) - x
        y = r.y + int(7 * u)
        c_main = C_TEXT if fits else C_TEXT_DIM
        c_sub = C_TEXT_DIM if fits else _shade(C_TEXT_DIM, 0.75)
        shown = wing_shown(name, self.lib)[0]
        in_slot = g.build.slot(g.sel).wing == name
        on_car = in_slot or any(g.build.slot(k).wing == name for k in SLOTS)
        tag = "in this slot" if in_slot else ("on the car" if on_car else "")
        tag_w = text.width(tag, 11) + int(8 * u) if tag else 0
        text.blit(screen, _cut(text, shown, wmax - tag_w, 14, bold=True), x, y, 14, c_main,
                  bold=True)
        if tag:
            text.blit(screen, tag, r.right - int(10 * u), y + int(2 * u), 11, ui.C_ACCENT, right=True)
        y += int(21 * u)
        role = "side wing" if w.role == "flank" else "top wing"
        area = float(w.legacy.get("S", w.S)) if w.legacy else float(w.S)
        ph = w.plate_h_flown
        if w.mount == "endplate":
            mount = f"endplates {ph:.2f} m carry it"
        else:
            mount = (f"plates {ph:.2f} m, " if ph > 0.0 else "no plates, ") + (
                "pylons" if w.mount == "pylon" else "no mount")
        rows = [(f"{role}  {'built in' if w.builtin else 'yours'}", c_sub),
                (f"span {w.span:.2f} m  area {area:.2f} m2", c_main),
                (origin_text(self.lib, name), c_sub),
                (mount, c_sub)]
        if fits and past:
            status, c_st = (f"fits in Unlimited: past the {lim:.2f} m limit (runs UNLIMITED)",
                            C_PANEL_ON)
        elif fits:
            status, c_st = ("fits (fitted now)" if in_slot else
                            f"fits the {'top' if w.role == 'top' else 'side'} slot: ENTER"), C_OK
        else:
            status, c_st = why, C_WARN
        if text.width(status, 12) > wmax and ": " in status:
            #  too long for a line: broken where it breaks, what and then the
            #  numbers ('too wide for the Opel Corsa:' / '1.88 m, the limit ...')
            a_, b_ = status.split(": ", 1)
            st_lines = (_wrap_lines(text, a_ + ":", wmax, 12, 2)
                        + _wrap_lines(text, b_, wmax, 12, 2))[:3]
        else:
            st_lines = _wrap_lines(text, status, wmax, 12, 3)
        lh = int(15 * u)
        y_st = r.bottom - int(7 * u) - len(st_lines) * lh
        for s_, c_ in rows:
            if y + int(15 * u) > y_st:
                break                               # the reason goes first
            text.blit(screen, _cut(text, s_, wmax, 12), x, y, 12, c_)
            y += int(16 * u)
        for j, ln in enumerate(st_lines):
            text.blit(screen, ln, x, y_st + j * lh, 12, c_st)
        self.drawn.append((name, fits, why, r))


class SavedCarsPage(_CardGrid):
    """SAVED CARS (task 46): every saved build as a card with a picture of
    its car carrying its wings -- GarageView's own `draw_scene` into a small
    offscreen surface, cached per build and redrawn only when the build, the
    wings it names, the car's paint or the card's size change (and dropped
    on a save, a rename or a delete: `invalidate`), at most
    PICTURES_PER_FRAME new pictures a frame. Its order is the library page's
    (`prerace.pick_order`: this car's builds, the any-car ones, then every
    other car's). `drawn`: [(name, status, rect)] as the last frame drew."""

    CARD_H = 300
    PIC_H = 164

    def __init__(self, g: "Garage"):
        super().__init__(3)
        self.g, self.lib = g, g.lib
        self._pics: dict = {}          # name -> (signature, Surface)
        self._views: dict = {}         # (car, size) -> GarageView on its own surface
        self.drawn: list = []
        self.rendered = 0              # pictures drawn by the last frame (the check reads it)
        self.refresh()

    def refresh(self, keep=None) -> None:
        from .prerace import pick_order
        self.set_names(pick_order(self.lib.builds, self.g.car), keep)

    def invalidate(self, name: str | None = None) -> None:
        """Drop the picture of build `name` (every picture when None)."""
        if name is None:
            self._pics.clear()
        else:
            self._pics.pop(name, None)

    def car_of(self, name: str) -> str:
        """The car build `name` belongs to: its tag when the game has that
        car, else ("" -- an any-car build, or a car taken out) this one's."""
        js = self.lib.builds.get(name)
        c = js.get("car", "") if isinstance(js, dict) else ""
        c = c if isinstance(c, str) else ""
        return c if _known_car(c) else ""

    def _paint(self, car: str):
        """The colour `car` is drawn in: the car page's own preview paint for
        the car in the garage; for another, the drive's Paint setting for it
        resolved (`Garage.paint_for`, which the drive hands in -- the garage
        never imports `render`, CONTRACT section 1); the stock C_PAINT in a
        bare garage, as its preview."""
        if car == self.g.car:
            return self.g.view.paint
        rgb = None
        pf = getattr(self.g, "paint_for", None)
        if callable(pf):
            try:
                rgb = pf(car)
            except Exception:                      # noqa: BLE001 -- a picture's colour
                rgb = None
        return tuple(int(c) for c in (rgb or C_PAINT))

    def _signature(self, name: str, js: dict, car: str, size) -> str:
        """What the picture depends on: the build, the geometry of the wings
        it names, the car, its paint and the size."""
        geo = {}
        for k in SLOTS:
            sl = (js.get("slots") or {}).get(k) if isinstance(js.get("slots"), dict) else None
            w = self.lib.wings.get(sl.get("wing")) if isinstance(sl, dict) else None
            if w is not None:
                geo[k] = (w.name, w.role, w.airfoil, w.span, w.chord, w.taper, w.twist_deg,
                          w.plate_h, w.mount, w.pylon_frac, w.plate_cant_deg, w.plate_blend,
                          w.plate_shape, w.plate_chord_ratio, w.plate_chord_follows, w.engine,
                          w.ride_h)
        return json.dumps([js, geo, car, list(self._paint(car)), list(size)], sort_keys=True,
                          default=str)

    def picture(self, name: str, size, budget: list):
        """Build `name`'s picture at `size` (w, h): the cached one when it is
        current, else drawn now while `budget[0]` lasts (None when it is not
        drawn yet -- the card shows a placeholder this frame)."""
        js = self.lib.builds.get(name)
        if not isinstance(js, dict):
            return None
        car = self.car_of(name) or self.g.car
        sig = self._signature(name, js, car, size)
        held = self._pics.get(name)
        if held is not None and held[0] == sig:
            return held[1]
        if budget[0] <= 0:
            return held[1] if held is not None else None
        budget[0] -= 1
        try:
            surf = self._render(js, car, size)
        except Exception as exc:                   # noqa: BLE001 -- a picture never stops the page
            print(f"garage: no picture of '{name}' ({type(exc).__name__}: {exc})")
            surf = pygame.Surface(size)
            surf.fill(C_BG)
        self._pics[name] = (sig, surf)
        self.rendered += 1
        return surf

    def _render(self, js: dict, car: str, size) -> "pygame.Surface":
        key = (car, tuple(size))
        view = self._views.get(key)
        if view is None:
            view = GarageView(pygame.Surface(size), car)
            self._views[key] = view
        view.set_paint(self._paint(car))
        cam = Orbit(*size)
        cam.fit(view.geo)
        cam.dist *= PICTURE_ZOOM
        cam.pitch = math.radians(PICTURE_PITCH_DEG)
        cam.target = np.array(cam.target0) * np.array([1.0, 1.0, PICTURE_LIFT])
        b = CarBuild.from_json(js)
        b.clamp(self.lib, car)                   # drawn fitted to its car, as it is driven
        view.draw_scene(b, self.lib, cam, 1.0, "")
        return view.screen.copy()

    def status(self, name: str) -> tuple:
        """What ENTER does with build `name` here, and its colour: load it
        (this car's, an any-car one), change to its car and load it (another
        car's, in the drive's garage), or why not (a bare garage, a
        challenge's car)."""
        from .prerace import car_label
        g = self.g
        c = self.car_of(name)
        if not c or c == g.car:
            return "ENTER: load it", C_OK
        if g.settings is not None and not g.car_fixed:
            return f"ENTER: change to the {car_label(c)} and load it", C_PANEL_ON
        if g.car_fixed:
            return f"the {car_label(c)}'s build: this challenge drives the {car_label(g.car)}", C_TEXT_DIM
        return f"the {car_label(c)}'s build: change car from the drive's garage", C_TEXT_DIM

    def draw(self, screen, text: ui.Text, u: float) -> None:
        from .prerace import car_label
        R = lambda x, y, w, h: pygame.Rect(int(x * u), int(y * u), int(w * u), int(h * u))  # noqa: E731
        g = self.g
        from .prerace import pick_order
        self.set_names(pick_order(self.lib.builds, g.car))
        hr = ui.panel(screen, R(12, 12, 1256, 40))
        x = hr.x + int(12 * u)
        x += text.blit(screen, "SAVED CARS", x, hr.y + int(11 * u), 16, ui.C_SECTION, bold=True)
        own = sum(1 for n in self.names if self.car_of(n) in ("", g.car))
        text.blit(screen, f"   {len(self.names)} builds: {own} for the {car_label(g.car)}, "
                  f"{len(self.names) - own} for other cars", x, hr.y + int(13 * u), 12, C_TEXT_DIM)
        text.blit(screen, _cut(text, f"now in the garage: the {car_long(g.car)}, '{g.build.name}'",
                               int(560 * u), 12),
                  hr.right - int(12 * u), hr.y + int(13 * u), 12, C_TEXT, right=True)
        area = R(12, 60, 1256, 622)
        cards = self.layout(area, int(self.CARD_H * u), int(8 * u))
        hover = self.hit(pygame.mouse.get_pos())
        self.drawn, self.rendered = [], 0
        budget = [PICTURES_PER_FRAME]
        if not self.names:
            text.blit(screen, "(no saved cars yet: S saves the car in the garage)", area.x + 12,
                      area.y + 12, 14, C_TEXT_DIM)
        #  the selected card's picture first, so the one being looked at is
        #  never the one waiting
        order = sorted(cards, key=lambda ir: ir[0] != self.idx)
        self._now = g.saved_as()               # the library build the car in hand is
        pics = {}
        for i, rr in order:
            size = (rr.w - int(16 * u), int(self.PIC_H * u))
            pics[i] = self.picture(self.names[i], size, budget)
        for i, rr in cards:
            self._card(screen, text, u, self.names[i], rr, i == self.idx, i == hover, pics.get(i))

    def _card(self, screen, text, u, name, r, sel, hover, pic) -> None:
        from .prerace import car_label
        g = self.g
        self._card_bg(screen, r, sel, hover)
        pr = pygame.Rect(r.x + int(8 * u), r.y + int(8 * u), r.w - int(16 * u), int(self.PIC_H * u))
        if pic is not None:
            screen.blit(pic, pr.topleft)
        else:
            pygame.draw.rect(screen, C_BG, pr)
            text.blit(screen, "drawing ...", pr.centerx, pr.centery - 7, 12, C_TEXT_DIM, centre=True)
        pygame.draw.rect(screen, ui.C_BORDER, pr, 1)
        js = self.lib.builds.get(name) or {}
        b = CarBuild.from_json(js)
        c = self.car_of(name)
        x, y = r.x + int(10 * u), pr.bottom + int(8 * u)
        wmax = r.w - int(20 * u)
        st = g.settings
        dflt = ""
        if st is not None and hasattr(st, "build_of"):
            try:
                dflt = st.build_of(c or g.car)
            except Exception:                      # noqa: BLE001
                dflt = ""
        tag = "(default)" if dflt == name else ""
        now = getattr(self, "_now", None) == name
        tag = (tag + ("  " if tag else "") + "in the garage now") if now else tag
        tag_w = text.width(tag, 11) + int(8 * u) if tag else 0
        text.blit(screen, _cut(text, name, wmax - tag_w, 14, bold=True), x, y, 14, C_TEXT, bold=True)
        if tag:
            text.blit(screen, tag, r.right - int(10 * u), y + int(2 * u), 11,
                      C_OK if dflt == name else ui.C_ACCENT, right=True)
        y += int(21 * u)
        import cars as _cars
        if c:
            whose = _cars.CAR_TITLES.get(c, c) + ("  (this car)" if c == g.car else "")
        else:
            raw = js.get("car", "") if isinstance(js.get("car", ""), str) else ""
            whose = (f"any car ({raw} is not in the game now)" if raw else
                     "any car (saved before builds knew their car)")
        text.blit(screen, _cut(text, whose, wmax, 12), x, y, 12, C_OK if c == g.car else C_TEXT_DIM)
        y += int(17 * u)
        wl = [("side" if b.mirror else "left", b.left.wing)]
        if not b.mirror:
            wl.append(("right", b.right.wing))
        wl.append(("top", b.top.wing))
        for lbl, wn in wl:
            shown = wing_shown(wn, self.lib)[0] if wn else "none"
            if wn and wn not in self.lib.wings:
                shown = f"{wn} (not in the library)"
            text.blit(screen, _cut(text, f"{lbl:<6s}{shown}", wmax, 12), x, y, 12,
                      C_TEXT if wn else C_TEXT_DIM)
            y += int(15 * u)
        status, c_st = self.status(name)
        try:
            over = bodies.over_limits(js, self.lib, c or g.car)
        except Exception:                          # noqa: BLE001 -- a tag is never worth a crash
            over = []
        lines = _wrap_lines(text, status, wmax, 12, 2)
        if over:
            lines = lines[:1] + [_cut(text, "UNLIMITED: " + bodies.limits_text(over), wmax, 12)]
        lh = int(15 * u)
        y_st = r.bottom - int(7 * u) - len(lines) * lh
        for j, ln in enumerate(lines):
            text.blit(screen, ln, x, y_st + j * lh, 12,
                      C_WARN if (over and j == len(lines) - 1) else c_st)
        self.drawn.append((name, status, r))


# =========================================================================== #
#  THE EDITOR LOOP                                                             #
# =========================================================================== #
# =========================================================================== #
#  STEP 1 -- THE MISSION PAGE                                                 #
# =========================================================================== #
class MissionPage:
    """What the wing is for, stated before anything is designed -- stage 1,
    AeroBO's Mission, and the one part of the design page that is carsim's
    own (the owner: "the only thing that could be different is the mission
    and the looks").

    The requirement is a LAP of one of carsim's circuits on a surface, for a
    slot. It supplies AeroBO's operating point (`aerobo_models.DesignSession
    .op`): the design speed is that lap's mean speed (or a typed one), the
    design CZ AeroBO's reference 1.0, the air carsim's. The Search & budget
    tab is AeroBO's `SearchPolicy` -- its measured budgets, or the player's.

    The page is GATED: nothing downstream opens until `state()` has been
    pressed. A mission nobody confirmed is a default.
    """

    def __init__(self, g: "Garage"):
        self.g = g
        self.result = None          # the car as it stands
        self.bare = None            # the same car with nothing on it
        self.err = ""
        #: `signature()` at the last `update()`, and the car it flew: the two
        #: laps cost ~2 x 28 ms, so a page that is merely re-shown does not
        #: re-fly them unless the circuit, the surface, the slot or the car moved
        self.updated_for = None
        self._car_for = None
        P = ui.Param
        pol = g.policy
        self.params = Form([
            P("m", "MISSION  (what the wing is for)", None, kind="label"),
            P("track", "job", lambda: self.mission().track, self._set_track, kind="choice",
              choices=list(ms.JOBS),
              help="a circuit: carsim's own geometry, the lap integrated over the arcs "
                   "and straights drive/track.py defines it with (the dragstrip is not "
                   "offered -- with no corner, every wing on it is pure drag). stopping: a "
                   "straight-line stop from the speed below, as the Stop from 100 challenge "
                   "(the top wing only). The side wings fly a circuit of their own"),
            P("surf", "surface", lambda: self.g.mission.surface, self._set_surface, kind="choice",
              choices=[n for n, _ in ms.MissionSpec.SURFACES],
              help="tyre grip scale. 'wet' is published (qss.sweep, 0.55/0.87); "
                   "'damp' is an estimate and track.py says so"),
            P("r", "THE CAR AS IT STANDS", None, kind="label"),
            P("lap", "lap", lambda: (self.result.time if isinstance(self.result, ms.LapResult) else 0.0), None,
              lo=None, hi=None, unit="s", fmt="{:.3f}", enabled=False,
              help="quasi-steady: corners at their steady speed, straights the "
                   "acceleration profile met by the braking profile. An OPTIMUM, "
                   "not a prediction -- a driven lap is slower"),
            P("dlap", "vs the car with no wings", lambda: self._delta(), None,
              lo=None, hi=None, unit="s", fmt="{:+.3f}", enabled=False,
              help="what the wings currently fitted are worth. Negative is faster"),
            P("vm", "mean speed", lambda: float(getattr(self.result, "v_mean", 0.0) or 0.0), None,
              lo=None, hi=None, unit="m/s", fmt="{:.1f}", enabled=False,
              help="length / time of this lap"),
            P("vx", "fastest point", lambda: float(getattr(self.result, "v_max", 0.0) or 0.0), None,
              lo=None, hi=None, unit="m/s", fmt="{:.1f}", enabled=False),
            P("tc", "in corners", lambda: float(getattr(self.result, "t_corner", 0.0) or 0.0), None,
              lo=None, hi=None, unit="s", fmt="{:.2f}", enabled=False,
              help="how much of the lap the wing's downforce or side force is paid for by"),
            P("ts", "on straights", lambda: float(getattr(self.result, "t_straight", 0.0) or 0.0), None,
              lo=None, hi=None, unit="s", fmt="{:.2f}", enabled=False,
              help="how much of it the wing's drag is charged over"),
            P("a", "ACTIONS", None, kind="label"),
            P("go", "state this mission  ->  design the section  (ENTER)", None,
              lambda _: self.g.state_mission(), kind="action"),
            #  -- appended for the AeroBO shell (the list above is pinned) --
            P("slot", "slot", lambda: ("left" if self.g.sel == "right" and self.g.build.mirror
                                       else self.g.sel), self._set_slot, kind="choice",
              choices=list(SLOTS),
              help="the car page's selection; the side pair is mirrored unless M split "
                   "them. Stating the mission opens the design stages for this slot"),
            P("dflt", "Published defaults", None, lambda _: self._defaults(), kind="action",
              help="arena, dry: the circuit and the grip every published number was "
                   "measured on. The mission has to be stated again"),
            P("vstop", "stop from", lambda: float(self.g.mission.v_stop_kmh), self._set_vstop,
              step=10.0, fine=5.0, lo=ms.V_STOP_BAND[0], hi=ms.V_STOP_BAND[1], unit="km/h",
              fmt="{:.0f}", enabled=lambda: self.mission().is_stop,
              help="the stop's start speed: the design speed, and where the stop is timed "
                   "from to a standstill. 100 is the Stop from 100 challenge's, 150 the Air "
                   "brake challenge's. One speed for the car: a side wing's stop (the air "
                   "brake) starts there too"),
            #  -- Search & budget: AeroBO's policy (aerobo_models.SearchPolicy) --
            P("smode", "where the budgets come from", lambda: pol.mode, self._set_smode,
              kind="choice", choices=list(pol.MODES),
              help="recommended: WingLab's measured plan (api.recommended_search over "
                   "search_budget.json, 1260 runs, 19 cases) -- 164 evaluations for each "
                   "section, 53 for the wing, at balanced. own: the three budgets are yours"),
            P("effort", "aim for", lambda: pol.effort, self._set_effort,
              kind="choice", choices=list(pol.EFFORTS),
              enabled=lambda: pol.recommended(),
              help="quick / balanced / thorough: WingLab's three measured budgets "
                   "(sections 109 / 164 / 240, the wing 42 / 53 / 87)"),
            P("stopconv", "stop when it stops improving",
              lambda: bool(pol.stop_when_converged), self._set_stop, kind="bool",
              help="WingLab's ConvergenceStop at the plan's patience and tolerance (the wing: "
                   "40 evaluations, 0.2 %); the budget stays the backstop. The section plans "
                   "have no adaptive rule"),
            P("own_af", "own budget: 2 Airfoil", lambda: int(pol.own["airfoil"]),
              self._set_own("airfoil"), kind="int", lo=4, hi=1000,
              enabled=lambda: not pol.recommended()),
            P("own_ep", "own budget: 2.8 Endplate", lambda: int(pol.own["plate"]),
              self._set_own("plate"), kind="int", lo=4, hi=1000,
              enabled=lambda: not pol.recommended()),
            P("own_w", "own budget: 3 Wing", lambda: int(pol.own["wing"]),
              self._set_own("wing"), kind="int", lo=4, hi=1000,
              enabled=lambda: not pol.recommended()),
            #  -- Design point: this slot's operating point --
            P("vauto", "design speed from the job", lambda: self._session().V_typed is None,
              self._set_vauto, kind="bool",
              help="on: the job's own speed -- a circuit lap's mean speed with this slot "
                   "empty (a side wing's own circuit), or a stop's start speed -- where "
                   "every coefficient is read and the sections' Reynolds numbers come "
                   "from. off: a speed you type"),
            P("vdes", "design speed", lambda: float(self._session().op.V), self._set_V,
              step=1.0, fine=0.25, lo=5.0, hi=90.0, unit="m/s", fmt="{:.1f}",
              enabled=lambda: self._session().V_typed is not None),
            P("cz", "design CZ", lambda: float(self._session().op.cz_design), self._set_cz,
              step=0.05, fine=0.01, lo=0.1, hi=3.0, fmt="{:.2f}",
              help="WingLab's reference lift coefficient of a car wing (1.0): the main "
                   "section is screened and designed at it. The plate's is 0"),
        ], title="STEP 1 of 2   MISSION")
        self.update()

    # -- rows ----------------------------------------------------------------
    def _session(self):
        return self.g.design_page.session_for(self.g.sel)

    def mission(self):
        """The mission the SELECTED slot flies: the top wing's is the car's;
        a side wing's is its own job in the same conditions (`MissionSpec.
        for_side`; the owner, 2026-09-26: "why no longer circuits?")."""
        m = self.g.mission
        return m.for_side() if SLOT_ROLE[self.g.sel] == "flank" else m

    def job_choices(self) -> list:
        """What the job select offers the selected slot: the circuits and the
        stop, to either wing (the owner, 2026-09-26: "Side wing should also
        have 'stopping' mission"; a side wing's stop is the air brake's)."""
        return list(ms.SIDE_JOBS if SLOT_ROLE[self.g.sel] == "flank" else ms.JOBS)

    def _set_track(self, v):
        m = self.g.mission
        if SLOT_ROLE[self.g.sel] == "flank":
            was = m.side_track
            m.side_track = str(v) if str(v) in ms.SIDE_JOBS else "arena"
            now = m.side_track
        else:
            was = m.track
            m.track = str(v) if str(v) in ms.JOBS else "arena"
            now = m.track
        m.stated = False
        self.update()
        if now != was:
            self.g.log(f"job set to {self.job_words()[0]} — the mission has to be "
                       f"stated again")

    def _set_vstop(self, v) -> None:
        lo, hi = ms.V_STOP_BAND
        was = self.g.mission.v_stop_kmh
        self.g.mission.v_stop_kmh = float(min(max(round(float(v)), lo), hi))
        if self.g.mission.v_stop_kmh != was:
            self.g.mission.stated = False
            self.update()
            self.g.log(f"stop from {self.g.mission.v_stop_kmh:.0f} km/h — the mission has to "
                       f"be stated again")

    def _set_surface(self, v):
        was = self.g.mission.mu_scale
        for nm, sc in ms.MissionSpec.SURFACES:
            if nm == str(v):
                self.g.mission.mu_scale = sc
        self.g.mission.stated = False
        self.update()
        if self.g.mission.mu_scale != was:
            self.g.log(f"surface set to {self.g.mission.surface} (grip × "
                       f"{self.g.mission.mu_scale:.2f}) — the mission has to be stated again")

    def _set_slot(self, v) -> None:
        """Which slot the design stages open for. Not a change of mission --
        the circuit and the surface are the car's -- but the stages belong to
        one slot, so stating the mission for another opens that one's."""
        if str(v) in SLOTS:
            self.g.sel = str(v)

    def _defaults(self) -> None:
        """Published defaults: arena, dry. One re-fly of the two laps, and
        the same log lines the two rows write when this moved either."""
        m = self.g.mission
        track, mu = m.track, m.mu_scale
        m.track = m.side_track = "arena"
        m.mu_scale = dict(ms.MissionSpec.SURFACES)["dry"]
        m.v_stop_kmh = ms.V_STOP_KMH
        m.stated = False
        self.update()
        if m.track != track:
            self.g.log(f"circuit set to {m.track} — the mission has to be stated again")
        if m.mu_scale != mu:
            self.g.log(f"surface set to {m.surface} (grip × {m.mu_scale:.2f}) — the mission "
                       f"has to be stated again")

    def _set_smode(self, v) -> None:
        if self.g.policy.set_mode("own" if str(v) == "own" else "recommended"):
            self.g.log("search settings: WingLab's measured plan is live — every budget is "
                       "its recommendation" if self.g.policy.recommended() else
                       "search settings: your own budgets are live — nothing moves them")

    def _set_effort(self, v) -> None:
        if self.g.policy.set_effort(str(v)):
            self.g.log(f"effort: {self.g.policy.effort} — the recommended budgets follow")

    def _set_stop(self, v) -> None:
        pol = self.g.policy
        if bool(v) == bool(pol.stop_when_converged):
            return
        pol.stop_when_converged = bool(v)
        self.g.log("stop when converged: on — a wing run ends once it stops improving"
                   if pol.stop_when_converged else
                   "stop when converged: off — every run spends its whole budget")

    def _set_own(self, which: str):
        def f(v):
            self.g.policy.own[which] = int(min(max(int(v), 4), 1000))
        return f

    def _set_vauto(self, v) -> None:
        s = self._session()
        s.set_V(None if bool(v) else s.op.V)

    def _set_V(self, v) -> None:
        self._session().set_V(float(v))

    def _set_cz(self, v) -> None:
        self._session().set_cz(float(v))

    def signature(self) -> tuple:
        """What stating the mission commits the design stages to: the job
        (the circuit, or the stop with its speed), the surface and the slot.
        A side wing's is its own circuit -- the top wing's job never moves it."""
        m = self.mission()
        return (m.job_key, m.surface, self.g.sel)

    def job(self) -> str:
        """The selected slot's job: STOPPING or the circuit's name (a side
        wing's own circuit, `mission()`)."""
        m = self.mission()
        return ms.STOPPING if m.is_stop else str(m.track)

    def job_words(self) -> tuple:
        """(the job, the surface) as the tree, the chips and the tags print
        them: "arena", "stopping 100 km/h"."""
        job, m = self.job(), self.mission()
        if job == ms.STOPPING:
            return (f"stopping {m.v_stop_kmh:.0f} km/h", m.surface)
        return (str(m.track), m.surface)

    def flies(self) -> bool:
        """Can the job be stated? Its lap or its stop has to close."""
        return self.result is not None and bool(self.result.ok)

    def _car_key(self) -> str:
        return json.dumps(self.g.build.to_json(), sort_keys=True, default=str)

    def _delta(self) -> float:
        """The wings as fitted against none: seconds on a lap, metres on a stop."""
        if self.result is None or self.bare is None:
            return 0.0
        if isinstance(self.result, ms.StopResult):
            return self.result.distance - self.bare.distance
        return self.result.time - self.bare.time

    # -- analysis ------------------------------------------------------------
    def profile(self):
        return self.mission().profile(make_track)

    def update(self) -> None:
        """Fly the job with the car as it stands and with no wings: the lap
        of a circuit (`LapResult`; a side wing's own circuit) or the stop
        (`StopResult`; a side wing's on the air brake, both panels out)."""
        self.err = ""
        try:
            job, mu = self.job(), self.g.mission.mu_scale
            if job == ms.STOPPING:
                m = self.mission()
                v0, n = m.v_stop, m.stop_flanks
                self.result = ms.stop(v0, self.g.build.mission_aero(self.lib), mu_scale=mu,
                                      flanks=n)
                self.bare = ms.stop(v0, ms.MissionAero(), mu_scale=mu, flanks=n)
            else:
                pf = self.profile()
                self.result = ms.lap(pf, self.g.build.mission_aero(self.lib), mu_scale=mu)
                self.bare = ms.lap(pf, ms.MissionAero(), mu_scale=mu)
        except Exception as exc:                      # a bad circuit never kills the page
            self.result = self.bare = None
            self.err = str(exc)
        self.updated_for = self.signature()
        self._car_for = self._car_key()
        #  the slot's operating point reads the job (its design speed): re-read
        #  it now, so the page quotes the new job's speed before it is stated
        dp = getattr(self.g, "design_page", None)
        s = dp.sessions.get(self.g.sel) if dp is not None and hasattr(dp, "sessions") else None
        if s is not None:
            s.refresh_op()
            if s.signature is None:                   # never opened: follow the job
                s.wing.default_objective(quiet=True)

    def stale(self) -> bool:
        """Would `update()` fly anything new? The circuit, the surface, the
        slot or the car as it stands moved since the last one."""
        return (self.result is None or self.signature() != self.updated_for
                or self._car_key() != self._car_for)

    @property
    def lib(self):
        return self.g.lib

    def state(self) -> None:
        """State it and open the design stages for the slot (`DesignPage.open`:
        the slot's session is kept if it was made for this circuit, surface
        and car). `Garage.state_mission` is the entry point that knows when a
        mission already stated need not be."""
        if not self.flies():
            self.g.say(f"this mission does not fly: {self.err or 'the lap did not close'}",
                       "warning")
            return
        self.g.mission.stated = True
        self.g.open_section()
        s = self.g.design_page.session
        r, job = self.result, self.job()
        v = s.op.V if s is not None else 0.0
        if job == ms.STOPPING:
            what = (f"stopping from {self.g.mission.v_stop_kmh:.0f} km/h "
                    + ("on the air brake " if self.mission().stop_flanks > 1 else "")
                    + f"({self.g.mission.surface}), {r.distance:.2f} m as the car stands "
                    f"({self._delta():+.2f} m vs no wings)")
        else:
            what = (f"{job} ({self.g.mission.surface}), lap {r.time:.3f} s as "
                    f"the car stands ({self._delta():+.3f} s vs no wings)")
        self.g.log(f"mission stated — {what}, design speed {v:.1f} m/s", "ok")


# =========================================================================== #
#  STEP 2 -- THE DESIGN PAGE: a navigator over the whole procedure            #
# =========================================================================== #
#: The procedure, written down, in the order AeroBO's own navigator carries
#: it. Each group is a thing being designed and each step is a stage of
#: designing it; the two SECTION groups run the same four stages because they
#: are the same problem pointed at two different surfaces.
#:
#: ENDPLATE is AeroBO's stage 2.8, and a real one: in the family the car
#: session flies ("car rear wing + endplates + free chord law") the plates are
#: a part with a section of their own, screened from AeroBO's SYMMETRIC
#: sections only and shape-optimised as a symmetric CST at cl 0 (the owner:
#: "endplates have to be symmetrical"). Plain fences -- Wing type's other
#: answer -- carry the wing's own section, and LOCK the stage.
#: The step labels are AeroBO's view labels -- the tree and the tabs of the
#: design shell show the same words (`design_shell.VIEW_META`).
DESIGN_TREE = (
    ("AIRFOIL   (the wing's own section)",
     (("af.screen", "Library screening"), ("af.rank", "Ranking"),
      ("af.section", "Section"), ("af.opt", "Shape optimisation"))),
    ("ENDPLATE  (the tip panels' section)",
     (("ep.screen", "Library screening"), ("ep.rank", "Ranking"),
      ("ep.section", "Section"), ("ep.opt", "Shape optimisation"))),
    ("WING",
     (("w.type", "Wing type"), ("w.box", "Design box"),
      ("w.solver", "Solver"), ("w.conv", "Convergence"))),
    ("RESULTS",
     (("r.summary", "Summary"), ("r.geometry", "Geometry"),
      ("r.loading", "Loading"), ("r.evals", "Evaluations"))),
)




class DesignPage:
    """Everything downstream of the mission, behind one navigator -- and,
    since the AeroBO pivot, nothing here designs anything: every stage is a
    view of the slot's `aerobo_models.DesignSession`, which calls AeroBO's own
    engine (PLAN2 §7.2).

    The left column is `DESIGN_TREE`, AeroBO's stages. The GATE is AeroBO's
    (`DesignSession.state`, PLAN2 D10): once the mission is stated, 2 Airfoil,
    2.8 Endplate and 3 Wing are all open -- the wing flies the family's own
    sections until one is chosen -- and 4 Results waits for a completed run.
    2.8 is LOCKED (not blocked) on a wing with plain fences: no work upstream
    opens it, only Wing type's designed endplates do.

    Sessions persist per slot in `sessions`. Stating an unchanged mission
    keeps a slot's session (screens, choices, runs); a changed circuit,
    surface or car (the other slots) invalidates it. TAB moves the focus
    between the navigator and the selected view's rows ("nav" / "rows",
    pinned); UP/DOWN drives whichever has it."""

    def __init__(self, g: "Garage"):
        self.g = g
        self.key = "left"
        self.sessions: dict = {}
        self.focus = "nav"
        #  the navigator is still a `ui.Nav` (its gate and `_rect` / `_hits` are
        #  pinned); `CaeTree` adds AeroBO's expand-on-select tree on top
        self.nav = CaeTree(DESIGN_TREE, title="DESIGN", state=self._state,
                           note=self._note, reason=self._reason)
        self.rank_list = ui.ListBox(title="RANKED AT THE SURFACE'S DESIGN POINT")
        self.msg = ""
        #: `mission_page.signature()` at the last `open`: an unchanged mission
        #: stated again re-validates instead of re-opening (AeroBO)
        self.opened_for: tuple | None = None
        self._rest_at_open = None                # `_car_rest()` at the last `open`
        self._slot_wing = None                   # the slot's wing at the last `open`

    # -- the session ------------------------------------------------------------------
    def session_for(self, key: str):
        """The slot's session, made on first use. A mirrored right flank is
        the left one's: the two carry one design."""
        am = _am()
        if key == "right" and self.g.build.mirror:
            key = "left"
        s = self.sessions.get(key)
        if s is None:
            #  no deck passed: the session reads the car's own (task 41,
            #  `g.car` -> bodies), and `g.unlimited` for its ceilings
            s = am.DesignSession(self.g, key, self.g.policy)
            self.sessions[key] = s
        return s

    @property
    def session(self):
        """The session of the slot this page is open for (None before the
        first `open`)."""
        k = "left" if (self.key == "right" and self.g.build.mirror) else self.key
        return self.sessions.get(k)

    @property
    def af(self):
        s = self.session
        return s.af if s is not None else None

    @property
    def ep(self):
        s = self.session
        return s.ep if s is not None else None

    @property
    def wing(self):
        s = self.session
        return s.wing if s is not None else None

    @property
    def results(self):
        s = self.session
        return s.results if s is not None else None

    def signature(self, key: str | None = None) -> tuple:
        """What a slot's session was made for: the job (the circuit, or the
        stop with its speed; a side wing's own circuit, whatever the top
        wing's job), the surface and the rest of the car (the slots it is
        scored beside)."""
        m, k = self.g.mission, key or self.key
        if SLOT_ROLE[k] == "flank":
            m = m.for_side()
        return (m.job_key, m.surface, self._car_rest(k))

    # -- opening --------------------------------------------------------------
    def open(self, key: str) -> None:
        """Open the stages for slot `key`: its session, kept when the circuit,
        the surface and the rest of the car are what it was made for, cleared
        (`DesignSession.invalidate`) when one of them moved."""
        #  a run still going belongs to the page being re-opened: it is
        #  abandoned, and nothing it found is applied
        if self.g.runs.busy:
            self.g.runs.cancel("the design page was re-opened")
        self.key = key
        self.session_for(key).open_for(self.signature(key))
        self.nav.refused = ""
        self.nav.select(self.nav.first_open(), force=True)
        self.focus = "nav"
        self.opened_for = self.g.mission_page.signature()
        self._rest_at_open = self._car_rest()
        self._slot_wing = self.g.build.slot(key).wing

    def _car_rest(self, key: str | None = None) -> str:
        """The part of the car this slot's wing is designed BESIDE but never
        edits: the top slot for a flank design, the flank pair for a top one
        (`CarBuild.mission_aero(exclude=...)`'s split). Only the car and
        library pages move it."""
        b = self.g.build
        keys = ("top",) if SLOT_ROLE[key or self.key] == "flank" else ("left", "right")
        return json.dumps([asdict(b.slot(k)) for k in keys], sort_keys=True, default=str)

    def car_moved(self) -> bool:
        """Has the car or library page changed the car UNDER this page since
        it opened -- the slot's wing swapped (W, a library pick, a reset) or
        the rest of the car the wing is designed beside? The page's own
        commits are not that."""
        if self.session is None:
            return True
        return (self.g.build.slot(self.key).wing != self._slot_wing
                or self._car_rest() != self._rest_at_open)

    def on_commit(self, key: str) -> None:
        """A wing was put in the slot from this page: the slot's wing is the
        page's own, not a change under it."""
        if key == self.key or (key == "left" and self.key == "right" and self.g.build.mirror):
            self._slot_wing = self.g.build.slot(self.key).wing

    def has_progress(self) -> bool:
        """Does the page hold work a re-open with a changed mission throws
        away -- a screen, a chosen section, a run? (The mission view warns
        before a re-state would.)"""
        s = self.session
        if s is None:
            return False
        return bool(s.af.ranked or s.ep.ranked or s.af.decision or s.ep.decision
                    or s.af.opt.get("report") or s.ep.opt.get("report") or s.wing.record)

    def goto(self, key: str) -> bool:
        """Show view `key`. With the AeroBO shell it is the shell's select
        (forced inside an unlocked stage); without one, the navigator's own
        gate, and a refusal is said rather than swallowed."""
        shell = getattr(self.g, "shell", None)
        if key.startswith("m."):
            self.g.goto_mission(key)
            return True
        if shell is not None:
            return bool(shell.select(key.split(".")[0], key))
        if self.nav.select(key):
            return True
        self.g.say(self.nav.refused, "info")
        return False

    # -- the gate ---------------------------------------------------------------
    def plate_locked(self) -> bool:
        """Is there a plate to design at all? AeroBO locks its endplate stage
        on a family whose plates are a FENCE (they carry the wing's chord and
        section): Wing type's 'endplates: fences'."""
        w = self.wing
        return bool(w is None or not w.choices["plates"])

    def _state(self, key: str) -> str:
        """The navigator's mark for view `key`: the stage's AeroBO state,
        mapped onto the Nav's four ("running" reads as open)."""
        s = self.session
        stage, _, view = key.partition(".")
        if s is None:
            return "blocked"
        st, _why = s.state(stage)
        if st == "locked":
            return "locked" if stage == "ep" else "blocked"
        if st == "error":
            return "blocked"
        if stage in ("af", "ep"):
            m = s.af if stage == "af" else s.ep
            return "done" if m.state(view) == "done" else "ready"
        if stage == "w":
            return "done" if (view == "conv" and s.wing.record) else "ready"
        return "done"

    def _reason(self, key: str) -> str:
        s = self.session
        stage, _, view = key.partition(".")
        if s is None:
            return "state the mission first: the design stages open for its slot"
        st, why = s.state(stage)
        if st in ("locked", "error"):
            return why
        if stage in ("af", "ep"):
            return (s.af if stage == "af" else s.ep).reason(view)
        return ""

    def _note(self, key: str) -> str:
        s = self.session
        if s is None:
            return ""
        stage, _, view = key.partition(".")
        if stage in ("af", "ep"):
            m = s.af if stage == "af" else s.ep
            if view == "rank" and not m.ranked:
                return "screen the library first (L)"
            if view == "section" and m.chosen:
                return m.chosen.get("name") or ""
            if view == "opt" and m.refusal():
                return "needs XFOIL" if "XFOIL" in (m.refusal() or "") else "needs the BO stack"
        if key == "r.summary" and not s.wing.record:
            return "no completed run yet"
        return ""

    def invalidate(self) -> None:
        """A form was rebuilt under us; the Design box form follows its
        family's rows (`WingModel.box_params` rebuilds on its own)."""
        if self.wing is not None:
            self.wing._box_form = None

    def settle(self) -> None:
        """Keep the cursor on a view that is still open: Wing type's fences
        switch LOCKS 2.8 under a cursor that may be standing in it."""
        if not self.nav.open(self.nav.current()):
            self.nav.select(self.nav.first_open(), force=True)

    def _model(self, key: str):
        return self.ep if key.startswith("ep.") else self.af

    # -- the rows the focused view owns ----------------------------------------
    def rows(self):
        """The Form the selected view binds to, or None (the ranking's table
        is `rank_list`, not a form)."""
        k = self.nav.current()
        if self.session is None:
            return None
        if k.startswith(("af.", "ep.")):
            m = self._model(k)
            if k.endswith(".screen"):
                return m.screen_params
            if k.endswith(".section"):
                return m.params
            if k.endswith(".opt"):
                return m.opt_params
            return None
        w = self.wing
        if k == "w.type":
            return w.type_params
        if k == "w.box":
            return w.box_params()
        if k == "w.solver":
            return w.solver_params
        if k == "w.conv":
            return w.conv_params
        if k.startswith("r."):
            return self.results.params
        return None

    # -- actions ---------------------------------------------------------------
    def act(self) -> None:
        """ENTER on the selected view."""
        k = self.nav.current()
        rows = self.rows()
        p = rows.current() if rows is not None else None
        if p is not None and p.kind == "action" and self.focus == "rows":
            p.activate()
            return
        #  a shut view can be ON SCREEN (the shell selects inside an unlocked
        #  stage by force, AeroBO's empty states): ENTER there says why
        if not self.nav.open(k):
            self.g.say(self._reason(k) or "this view is not open yet", "info")
            return
        if self.session is None:
            return
        if k.endswith(".screen"):
            self.screen()
        elif k.endswith(".rank"):
            m = self._model(k)
            if self.g.runs.job_for(m) is not None:
                self.g.toast("a run of this surface is going — stop it first", "warning")
                return
            m.use_ranked(self.rank_list.idx)
            self.msg = m.msg
        elif k.endswith(".opt"):
            self.optimise()
        elif k == "w.solver":
            self.run_wing()
        elif k == "w.conv":
            self.run_wing(extend=True)
        elif p is not None:
            p.activate()

    def optimise(self, extend: bool = False) -> None:
        """What O / SQUARE does (K: `extend`, Keep going) on the selected
        stage: the section's shape search, or the wing's run."""
        k = self.nav.current()
        if self.session is None:
            return
        if k.startswith(("af.", "ep.")):
            m = self._model(k)
            m.start_optimise(extend=extend)
            self.msg = m.msg
        else:
            self.run_wing(extend=extend)

    def run_wing(self, extend: bool = False) -> bool:
        """Launch the wing run LIVE and show its convergence (AeroBO)."""
        w = self.wing
        if w is None:
            return False
        ok = w.start_run(extend=extend)
        if ok:
            self.goto("w.conv")
        self.msg = w.msg
        return ok

    def screen(self) -> None:
        """What L does -- on the section stage it belongs to, and nowhere
        else: a key that acts on something off screen is the same defect as
        a gate that lets you skip a step."""
        k = self.nav.current()
        if not k.startswith(("af.", "ep.")) or self.session is None:
            self.msg = ("L screens a SECTION library -- select a view under 2 Airfoil or "
                        "2.8 Endplate first")
            return
        m = self._model(k)
        m.start_screen()
        self.msg = m.msg
        self.settle()

    def fit_current(self) -> str:
        """What F does (PLAN2 §7.3): take the highlighted ranked section on a
        ranking or section view, the optimised one on Shape optimisation, the
        family's own plate on 2.8's section view -- then move on to the stage
        it opened. Returns the section's name ("" when nothing was taken)."""
        k = self.nav.current()
        if not k.startswith(("af.", "ep.")) or self.session is None:
            self.msg = ("F takes a SECTION -- select a view under 2 Airfoil or 2.8 "
                        "Endplate first")
            return ""
        m = self._model(k)
        if k.endswith(".opt"):
            m.use_optimised()
        elif k == "ep.section" and not m.ranked:
            m.decline()
        else:
            m.use_ranked(self.rank_list.idx if k.endswith(".rank") else m.highlight)
        self.msg = m.msg
        if not m.finished():
            return ""
        self.advance(m.stage)
        return (m.chosen or {}).get("name") or ""

    def advance(self, group: str) -> None:
        """Step to the first open view of the stage AFTER `group`."""
        order = ["af", "ep", "w", "r"]
        try:
            i = order.index(group[:2].rstrip("."))
        except ValueError:
            return
        for nxt in order[i + 1:]:
            for key in self.nav.keys:
                if key.startswith(nxt + ".") and self.nav.open(key):
                    self.goto(key)
                    return
        self.settle()

    # -- the ranking table ----------------------------------------------------------
    def refresh_rank(self) -> None:
        """The ranking of the selected surface as `rank_list` rows: the name,
        each weighted criterion's value (+ its points), the composite."""
        k = self.nav.current()
        m = self._model(k)
        if m is None:
            self.rank_list.set_items([])
            return
        w = {c: float(v) for c, v in m.weights.items()}
        items = []
        for r in m.ranked:
            met, sc = r.get("metrics") or {}, r.get("scores") or {}
            bits = []
            for key, head, fmt in m.columns():
                v = met.get(_am().CRITERION_METRIC[key])
                if v is None:
                    continue
                try:
                    cell = f"{head} " + fmt.format(float(v))
                except (TypeError, ValueError):
                    continue
                if w.get(key, 0.0) > 0.0 and sc.get(key) is not None:
                    cell += f"({float(sc[key]):+.1f})"
                bits.append(cell)
            val = f"{r['composite']:.2f}" if r.get("composite") is not None else "-"
            items.append((r["name"], "  ".join(bits), val))
        keep = (m.chosen or {}).get("name") if m.decision == "library" else None
        self.rank_list.set_items(items, keep=keep)

    def draw(self, screen, text: ui.Text, plot: ui.Plot, u: float) -> str:
        """The page is drawn by the AeroBO shell (`design_shell.DesignShell`):
        the tree, the view and the chrome come from the models there. Kept,
        with its signature, for the callers that still ask; it draws nothing
        and has no help line to give."""
        return ""


class Garage:
    """Owns the window, the build, the library and the input. `run()`
    returns 'drive', 'title' (the menu's Main menu: the drive shows the
    title screen again), 'car' (the menu's Change car: the drive opens the
    garage again on `self.car_wanted`) or 'quit'; the (clamped) build is
    `self.build` every way (`self.design` is the same object, for the old
    attribute name)."""

    PAD_MOVE_X = 0.9      # m/s of x at full stick
    PAD_MOVE_H = 0.6      # m/s of h at full stick
    PAD_ORBIT = 2.2       # rad/s at full stick
    PAD_REPEAT = 0.14     # s between stick-held adjustments on the list pages

    def __init__(self, size=(1280, 800), build=None, pad=None, headless: bool = False,
                 lib: Library | None = None, car: str = "corsa", settings=None):
        import cars as _cars
        #: task 41: the car being driven (a `cars.py` key) -- its body, slot
        #: bands, span limits and default build -- and the drive's Settings
        #: (None in a bare garage: 'real' limits, no per-car defaults). A key
        #: `drive/bodies.py` draws but `cars.py` does not carry yet (the
        #: Express, the bus) is fitted on its body alone.
        self.car = (car if isinstance(car, str) and (car in _cars.CARS or car in bodies.STYLE_OF)
                    else "corsa")
        self.settings = settings
        #: the car the menu's Change car picked (a `cars.CAR_ORDER` key) when
        #: run() returns 'car'; None otherwise. `car_fixed`: the drive says
        #: the car is not the player's to change here (a challenge's car)
        self.car_wanted: str | None = None
        self.car_fixed = False
        #: task 46: with `car_wanted`, the saved build the drive opens that car
        #: with (the SAVED CARS page's ENTER on another car's build); None --
        #: the menu's Change car -- opens it with its own default
        self.build_wanted: str | None = None
        #: an action a page asked the loop for outside a key's return (SAVED
        #: CARS' change of car, from a click or the pad too): `frame` hands it on
        self._pending_action: str | None = None
        #: the card pages' hint lines as the last frame drew them (the check)
        self.saved_hint_drawn: list = []
        #: task 46: car key -> the RGB the drive's Paint setting draws it in
        #: (SAVED CARS' pictures of other cars); the drive sets it, None in a
        #: bare garage (the stock colour)
        self.paint_for = None
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        self.screen = pygame.display.set_mode(size)
        from . import branding
        pygame.display.set_caption(branding.caption("garage"))
        self.lib = lib or library()
        if isinstance(build, WingDesign):
            build = CarBuild.from_json(asdict(build))
        self._held = None                 # (the build handed in, its fitted JSON)
        #: where the per-map memory a rename updates lives (records.RECORDS_DIR;
        #: a self-check points it at a scratch folder)
        from .records import RECORDS_DIR
        self.records_root = RECORDS_DIR
        self.build: CarBuild = self._fit_in(build or CarBuild.load()
                                            or CarBuild.for_car(self.car))
        self.view = GarageView(self.screen, self.car)
        self.cam = Orbit(*self.screen.get_size())
        self.cam.fit(self.view.geo)
        self.text = self.view.text
        self.plot = ui.Plot(self.text)
        self.pad = pad
        self.headless = headless
        self.deploy_cmd = 0.0
        self.deploy = 0.0
        self.hint = ""                    # a property: the write stamps `_hint_t`
        #  round 3: the guide hint 'Try ready-made wings' wrote (`try_ready_made`):
        #  a background note (the wing data arriving) does not cover it while it shows
        self._hint_guide = ""
        #  task 45: the destructive press waiting for its second (`_confirm`):
        #  None, or {what, t (self._t_alive), keys that may confirm it, hint};
        #  the keys held down now (a held key's auto-repeat never confirms);
        #  and the car as it was before the last reset, for U (one level)
        self._armed: dict | None = None
        self._keys_down: set = set()
        self._undo_build: dict | None = None
        self._drag = False
        self._pad_prev: dict[str, bool] = {}
        self._pad_seeded = False      # first poll takes the live buttons as
        #                               'already down': the CROSS that chose
        #                               'Garage' in the drive's menu must not
        #                               be the CROSS that drives straight back
        self._t_alive = 0.0           # 'drive' is ignored in the first 0.35 s
        self._frames = 0
        self.menu = Menu("GARAGE")
        self._menu_stick = StickNav()
        self._nav_stick = StickNav()
        self._rep_t = 0.0
        self.page = "car"
        self.sel = "left"
        #  THE LIVE RUNS (drive/design_jobs.py): AeroBO's engine on ONE worker
        #  thread at a time, drained by the frame; what the jobs -- and the
        #  models -- have to say goes to the Output log and the toasts
        self.notices = dj.Notices()
        self.runs = dj.RunManager(self.notices)
        #  stage 1's Search & budget: AeroBO's measured plan or the player's
        #  own budgets, one policy for every slot (aerobo_models.SearchPolicy);
        #  `search` is the old dict spelling of the same object
        self.policy = _am().SearchPolicy()
        self.search = _SearchView(self.policy)
        #  AeroBO's BO stack (torch) is imported NOW, while the garage loads:
        #  the first view that states a search would otherwise pay the import
        #  inside a frame (aerobo_models.warm_up)
        _am().warm_up()
        #  where AeroBO's runs write (its `results_dir`) and carsim keeps their
        #  records: runs/aerobo/<slot>/. None writes nothing (the checks)
        self.aerobo_dir = os.path.join("runs", "aerobo")
        self.mission_tab = "m.operating"          # the mission page's remembered tab
        #  where exports are written -- the self-checks point it at a temp dir
        self.export_dir = os.path.join("runs", "export")
        #  the three design steps, in the order AeroBO walks them. The mission
        #  is held on the GARAGE and not on a page, because the section and the
        #  wing are both scored against it and a mission that lived on one page
        #  could be changed underneath the other.
        self.mission = ms.MissionSpec()
        #  the design page before the mission page: the mission's Design point
        #  rows read the slot's session
        self.design_page = DesignPage(self)
        self.mission_page = MissionPage(self)
        self.af_page = AirfoilPage(self)
        self.lib_page = LibraryPage(self)
        #  task 46: the two card pages, SAVED WINGS (L) and SAVED CARS (G)
        self.wings_page = SavedWingsPage(self)
        self.cars_page = SavedCarsPage(self)
        self.prompt = ui.TextPrompt()
        self._prompt_kind = ""
        self._af_return = "car"
        self.status = ""
        self.status_shown = ""            # what the bar's corner shows of it now
        #  the wing-design tutorial (drive/wing_tutorial.py, task 24): a
        #  WingTutor the frame and the menu query, or None; `progress` is the
        #  player's runs/progress.json (drive/progress.py), None in a script
        self.tutor = None
        self.progress = None
        #  THE AeroBO SHELL: the frame of the mission and design pages (last,
        #  because it reads every page above). It sets the widget kit's scale
        #  for this window and warms the fonts and the optimiser's imports.
        self.shell = DesignShell(self)

    # -- the hint (task 45) ------------------------------------------------
    #  Every write of `hint` stamps `_hint_t` on the garage's own clock
    #  (`_t_alive`), so the car page can let it go HINT_S later: a write IS
    #  something the player just did, and the same words written again (B
    #  with no builds, twice) show again. What only keeps the standing hint
    #  goes through `_note`, which leaves it and its clock alone.
    @property
    def hint(self) -> str:
        return self._hint

    @hint.setter
    def hint(self, s: str) -> None:
        self._hint = s
        self._hint_t = getattr(self, "_t_alive", 0.0)

    def hint_alpha(self) -> float:
        """How much of the hint is left to see: 1 for its first HINT_S -
        HINT_FADE_S seconds, down to 0 at HINT_S and after; 0 with none."""
        if not self._hint:
            return 0.0
        left = HINT_S - (self._t_alive - self._hint_t)
        return max(0.0, min(1.0, left / HINT_FADE_S))

    def _note(self, *msgs) -> None:
        """The first of `msgs` that says something becomes the hint; when
        none does, the standing hint is left as it is, clock and all."""
        for m in msgs:
            if m:
                self.hint = m
                return

    # the old attribute name
    @property
    def design(self) -> CarBuild:
        return self.build

    def set_paint(self, rgb) -> None:
        """The player's paint on the preview car (GarageView.set_paint):
        drive.drive calls it right after constructing the Garage, with the
        fitted car's paint resolved to an RGB. Cosmetic only."""
        self.view.set_paint(rgb)

    @property
    def unlimited(self) -> bool:
        """Settings' Wing limits is 'unlimited' (task 41): the editors may take
        a span past the car's physical limit (`bodies.span_ceiling`)."""
        return getattr(self.settings, "wing_limits", "real") == "unlimited"

    # -- pause menu (ESC / OPTIONS) -------------------------------------------
    def _menu_open(self, at: str = "") -> None:
        """The pause menu, its cursor on the row whose action is `at` (the
        first row when none is)."""
        secs = [("KEYBOARD", GARAGE_HELP_KB)]
        note = ""
        if self.pad is not None:
            secs.append(("PS5 DUALSENSE" if self.pad.layout == "ps" else "GAMEPAD",
                         GARAGE_HELP_PAD))
        else:
            note = "No gamepad connected - plug one in any time."   # input.MENU_NO_PAD's words
        tut_rows = []
        if self.tutor is not None and self.tutor.active:
            tut_rows = self.tutor.menu_rows()
        elif self.progress is not None:
            from .wing_tutorial import menu_row
            tut_rows = [menu_row(self.progress, self.tutor)]
        #  task 45: 'Main menu' is the way back to the title (a garage opened
        #  from it had none), and Quit says where it goes and asks twice.
        #  Change car (2026-09-27): only in the drive's garage -- the drive
        #  opens the next one on the picked car (a bare garage has no
        #  Settings) -- and not on a challenge's car (`car_fixed`)
        from .prerace import car_label
        car_row = ([(f"Change car  (now: {car_label(self.car)})", "cars")]
                   if self.settings is not None and not self.car_fixed else [])
        #  task 46: the two card pages took the text library's row and
        #  'Load a build ...' (Saved cars loads them; SHIFT+L still opens the
        #  text list), and the selected slot's wing is saved from here too --
        #  the pad's K (its prompt offers a free name, CROSS takes it)
        what = "top" if self.sel == "top" else "side"
        items = ([("Resume", "resume")] + car_row
                 + [(f"Design a wing for the {self.sel} slot", "design")]
                 + tut_rows
                 + [("Airfoil library", "airfoils"),
                    ("Saved wings", "saved_wings"),
                    ("Saved cars", "saved_cars"),
                    (f"Save the {what} wing as ...", "wing_save")]
                 + self._build_rows()
                 + [(self._row_label("defaults"), "defaults"),
                    #  (no 'Reset camera' row since Change car came: C here and on
                    #  the car page, R3 on the pad -- the footer says C -- and the
                    #  menu keeps its rows, which fit only as many; review 1)
                    ("Drive this car", "drive"),
                    ("Main menu", "title"),
                    (self._row_label("quit"), "quit")])
        self.menu.show(
            items=items, sections=secs, note=note, title="GARAGE",
            subtitle=self.build.summary(self.lib)[:120],
            footer="ESC / OPTIONS resume   R R reset car   C camera   ENTER / CROSS select",
            idx=next((i for i, (_, a) in enumerate(items) if a == at), 0))
        self._menu_stick = StickNav()

    def _row_label(self, act: str, lbl: str = "") -> str:
        """A menu row's label: for the reset and quit rows, what a second
        select does while it is armed; any other row keeps `lbl`."""
        armed = (self._armed is not None and act in MENU_ARMS
                 and self._armed["what"] == MENU_ARMS[act])
        again = "ENTER" + (" / CROSS" if self.pad is not None else "") + " again"
        if act == "defaults":
            return f"Reset car: {again} to remove all wings" if armed else "Reset car to defaults"
        if act == "quit":
            return f"Quit to desktop: {again}" if armed else "Quit to desktop"
        return lbl

    # -- two presses for what cannot be taken back (task 45) ----------------
    def _confirm(self, what: str, hint: str, keys) -> bool:
        """True on the press that CONFIRMS `what`: it was armed by the press
        before, less than ARM_S ago (the arm is spent; the caller does it).
        Otherwise `what` is armed, `hint` says what the second press will do,
        and False. `keys` may confirm it; any other key disarms (`_handle`)."""
        a = self._armed
        if a is not None and a["what"] == what and self._t_alive - a["t"] <= ARM_S:
            self._disarm()
            return True
        self._armed = dict(what=what, t=self._t_alive, keys=tuple(keys), hint=hint)
        if hint:
            self.hint = hint
        return False

    def _disarm(self) -> None:
        """Drop the arm; its 'again' hint and the menu row's label go with it."""
        a, self._armed = self._armed, None
        if a is None:
            return
        if a["hint"] and self.hint == a["hint"]:
            self.hint = ""
        if self.menu.open:
            self.menu.items = [(self._row_label(act, lbl), act) for lbl, act in self.menu.items]

    def _reset_car(self) -> None:
        """Every slot empty at the car's default station (R R, or the menu's
        row twice). The car as it was is kept for U, one level deep; saved
        builds are not touched."""
        had = self.build.has_any(self.lib)
        self._undo_build = self.build.to_json()
        self.build.reset(self.car)
        self.build.clamp(self.lib, self.car)
        self.hint = ("car reset: no wings - U puts them back" if had
                     else "car reset: slots at their default stations - U undoes it")

    def _undo_reset(self) -> None:
        """U on the car page: the car as it was before the last reset, in
        place (the reset is in place too), clamped. One level: spent here."""
        if self._undo_build is None:
            self.hint = "nothing to undo (U puts the wings back after a car reset)"
            return
        back = CarBuild.from_json(self._undo_build)
        for f in fields(CarBuild):
            setattr(self.build, f.name, getattr(back, f.name))
        self._undo_build = None
        self.build.clamp(self.lib, self.car)
        self.hint = "wings back" if self.build.has_any(self.lib) else "car back as it was"

    def _build_rows(self) -> list:
        """The pause menu's build rows (task 41): the pad's way to what S,
        SHIFT+S, B and F do on the car page, whose buttons are all taken."""
        #  task 45: 'Set as Express default' (it was 'Make it the Express
        #  default', 5 characters more) -- the widest row. With the wing
        #  tutorial's three rows the list scrolls, its 'v 16 more' goes beside
        #  the widest row, and the help beside them wrapped and ran past the
        #  footer at 1280x720, 1440x900 and 1600x900 (`_check_menu_fits`)
        #  Task 46: 'Load a build ...' is the menu's Saved cars row now (its
        #  action `build_load` still opens that page, for an old caller)
        from .prerace import car_label
        n, dflt = self.build.name, self.default_name()
        return [(f"Save build  '{_menu_name(n)}'" + ("" if self._own_build(n) else "  (asks a name)"),
                 "build_save"),
                ("Save build as a new name ...", "build_save_as"),
                (f"Set as {car_label(self.car)} default  (now: {_menu_name(dflt) or 'none'})",
                 "build_default")]

    def _car_menu(self) -> None:
        """The menu's CHANGE CAR list: every car, this one marked and under
        the cursor. A pick ends the garage ('car'); the drive opens the next
        one on that car with what its Settings Car row opens it with (its
        default build, ...), a build in hand that is in no library file saved
        there first (drive._car_build, _autosave_build)."""
        import cars as _cars
        items = [(_cars.CAR_TITLES.get(k, k) + ("  (this car)" if k == self.car else ""),
                  "car:" + k) for k in _cars.CAR_ORDER if k in _cars.CARS]
        items.append(("Back", "car_back"))
        self.menu.show(items=items, title="CHANGE CAR", sections=[], note="",
                       subtitle="each car opens with its own default build (F); "
                                "unsaved wings are kept in the library",
                       footer="ENTER / CROSS choose   ESC / OPTIONS back",
                       idx=next((i for i, (_, a) in enumerate(items)
                                 if a == "car:" + self.car), 0))
        self._menu_stick = StickNav()

    def _first_wing_choice(self) -> bool:
        """The first time a player opens the designer, ask ONCE: the guided
        first wing (drive/wing_tutorial.py) or straight to the designer. Not
        asked in a script (no progress), while the tutorial runs, or once it
        has been started or finished; the answer is kept in the progress file
        ('wing_tutorial'.offered). True when the choice is on screen."""
        if self.progress is None or (self.tutor is not None and self.tutor.active):
            return False
        from .wing_tutorial import SECTION, saved_state
        sv = saved_state(self.progress)
        if sv["done"] or sv["step"] or self.progress.section(SECTION).get("offered"):
            return False
        self.menu.show(items=[("Guided first wing (recommended)", "wt_first"),
                              ("Straight to the designer", "design_now")],
                       title="YOUR FIRST WING", sections=[], note="",
                       subtitle="the wing tutorial walks you through designing one, step by step",
                       footer="ENTER / CROSS choose   ESC / OPTIONS back")
        self._menu_stick = StickNav()
        return True

    def _first_wing_answered(self) -> None:
        if self.progress is not None:
            from .wing_tutorial import SECTION
            self.progress.section(SECTION)["offered"] = True
            self.progress.save(SECTION)

    def _menu_action(self, action: str | None) -> str | None:
        """Run a menu action; returns 'drive' / 'title' / 'car' / 'quit' for
        the loop, else None.

        While a design run is live every action but resume, drive, main menu,
        change car and quit is refused (it would change the car or the page
        under the run); leaving -- another car is leaving too -- abandons the
        run first: nothing it found is applied."""
        if action is None:
            return None
        if self._armed is not None and self._armed["what"] != MENU_ARMS.get(action):
            self._disarm()                 # the menu left, or another row chosen
        if action == "resume":
            if self.menu.title == "CHANGE CAR":        # its ESC / CIRCLE: the menu again
                self._menu_open(at="cars")
            return None
        if action == "cars":
            self._car_menu()
            return None
        if action == "car_back":
            self._menu_open(at="cars")
            return None
        if action.startswith("car:"):
            from .prerace import car_label
            want = action[4:]
            if want == self.car:
                self.hint = f"already the {car_label(self.car)}"
                return None
            self._leave_runs()
            self.car_wanted = want
            return "car"
        if self.runs.busy and action not in ("drive", "title", "quit"):
            self.say("a run is in progress — stop it first", "warning")
            return None
        if action == "defaults":
            #  task 45: the first select (or R) only arms it -- the menu comes
            #  back with the row saying what a second one does, cursor on it
            if self._confirm("reset", "", MENU_CONFIRM_KEYS):
                self._reset_car()
            else:
                self._menu_open(at="defaults")
        elif action == "camera":
            self.cam.reset()
        elif action == "design":
            self.open_mission()
        elif action == "design_now":                 # the first-wing choice: no tutorial
            self._first_wing_answered()
            self.open_mission()
        elif action == "airfoils":
            self.open_airfoils()
        elif action == "library":
            self.open_library()
        elif action == "saved_wings":                 # task 46
            self.open_saved("wings")
        elif action in ("saved_cars", "build_load"):
            self.open_saved("cars")
        elif action == "wing_save":
            #  a free name: a pad can only take the name offered (CROSS)
            self.prompt_wing(as_new=True)
        elif action == "build_save":
            self.save_build()
        elif action == "build_save_as":
            self.save_build(as_new=True)
        elif action == "build_default":
            self.make_default()
        elif action in ("wt_start", "wt_resume", "wt_first"):
            self._first_wing_answered()
            from .wing_tutorial import WingTutor, saved_state
            start = saved_state(self.progress)["step"] if action == "wt_resume" else None
            self.tutor = WingTutor(self.progress, start=start)
            self.hint = f"wing tutorial: {self.tutor.label()}"
            if action == "wt_resume":
                #  task 45: 'continue at step N' goes on AT step N, on its
                #  page, the box and the hint saying it -- or, when that
                #  step's work went with the garage it was done in, from the
                #  nearest step that can be done, and they say so
                self.tutor.resume(self)
            if action == "wt_first":
                #  task 45: the first-wing choice came from a D (L3, the
                #  menu's Design row) -- the press step 1 asks for is the
                #  one just made, so it runs, and step 1 passes by itself.
                #  The guided wing is a FLANK wing: the top slot gives way
                #  to the left one
                self.open_mission("left" if self.sel not in ("left", "right") else None)
        elif self.tutor is not None and self.tutor.menu_action(action):
            #  task 45: 'Step 3 of 10: ...', the box's own words (it said the
            #  title alone)
            self.hint = "wing tutorial: " + ("ended" if not self.tutor.active
                                             else self.tutor.label())
        elif action == "quit":
            #  task 45: the first select arms it -- the menu comes back with
            #  the row asking again, cursor on it; the second leaves the game
            if self._confirm("quit", "", MENU_SELECT_KEYS):
                self._leave_runs()
                return action
            self._menu_open(at="quit")
        elif action in ("drive", "title"):
            self._leave_runs()
            return action
        return None

    def _leave_runs(self) -> None:
        """Leaving the garage abandons a live design run: nothing it found
        is applied."""
        if self.runs.busy:
            self.runs.cancel("left the garage")

    # -- the Output log and the toasts ------------------------------------------
    def log(self, text: str, level: str = "info") -> None:
        """A line for the Output log (info / ok / warn / error)."""
        self.notices.log(text, level)

    def toast(self, text: str, kind: str = "info") -> None:
        """A toast (positive / negative / warning / info)."""
        self.notices.toast(text, kind)

    def say(self, text: str, kind: str = "info") -> None:
        """A refusal or a notice the player should see NOW: a toast on the
        AeroBO shell pages, the old hint line everywhere else."""
        if self.page in ("mission", "section") and getattr(self, "shell", None) is not None:
            self.toast(text, kind)
        else:
            self.hint = text

    # -- pages ------------------------------------------------------------------
    # -- the three design steps, in order ------------------------------------
    def open_mission(self, key: str | None = None) -> None:
        """STEP 1. Every route into designing a wing comes through here."""
        if key is not None:
            self.sel = key
        if self._first_wing_choice():
            return
        self.mission_page.update()
        #  the slot's design session is made HERE, beside the mission's own
        #  laps, and not by the first frame whose tree chips ask for it: its
        #  operating point flies a lap too (40-60 ms), which put the first
        #  mission frame over its budget
        self.design_page.session_for(self.sel)
        self.page = "mission"

    def goto_mission(self, tab: str = "m.operating") -> None:
        """Show the mission page on `tab` WITHOUT a commit or a reset (the
        shell's tree). The two laps it flies (~2 x 28 ms) are flown again only
        when the circuit, the surface, the slot or the car has moved."""
        self.mission_tab = tab
        if self.mission_page.stale():
            self.mission_page.update()
        self.page = "mission"

    def state_mission(self) -> bool:
        """THE one entry point that states the mission (`go`, the tool bar's
        Run, ENTER on the mission page).

        AeroBO's accept is idempotent: stating a mission that is already
        stated -- the design page open for this slot, and the circuit, the
        surface and the slot what it was opened for -- only re-checks it, and
        the design stages keep their progress. Anything else opens the slot's
        stages (`DesignPage.open`), which keeps the slot's session when it was
        made for this circuit, surface and car and clears it when not."""
        dp, mp = self.design_page, self.mission_page
        if self.runs.busy:
            self.toast("a run is in progress — stop it first", "warning")
            return False
        if self.mission_restate_keeps():
            if mp.stale():
                mp.update()
            if not mp.flies():
                self.say(f"this mission does not fly: {mp.err or 'the lap did not close'}",
                         "warning")
                return False
            self.log("mission already stated — nothing changed; the design stages keep "
                     "their progress")
            shell = getattr(self, "shell", None)
            if shell is not None:
                shell.select("af", "af.screen")
            else:
                dp.nav.select("af.screen", force=True)
                self.page = "section"
            return True
        mp.state()
        return bool(self.mission.stated and self.page == "section")

    def mission_restate_keeps(self) -> bool:
        """Would stating the mission NOW only re-check it (`state_mission`'s
        idempotent path)? Stated, the design page open for this slot, the
        circuit, surface and slot what it was opened for, and the car not
        moved under it."""
        dp = self.design_page
        return bool(self.mission.stated and dp.session is not None and dp.key == self.sel
                    and dp.opened_for == self.mission_page.signature() and not dp.car_moved())

    def slot_chord(self, key: str | None = None):
        """The root chord of the wing in slot `key` (the library's), or None
        for an empty slot."""
        key = self.sel if key is None else key
        spec = self.lib.wings.get(self.build.slot(key).wing)
        return float(spec.chord) if spec is not None else None

    def open_section(self, key: str | None = None) -> None:
        """STEP 2. The DESIGN page -- the navigator over everything downstream
        of the mission. Gated on a STATED mission."""
        if key is not None:
            self.sel = key
        if not self.mission.stated:
            self.say("state the mission first: nothing downstream can be judged without one")
            self.open_mission()
            return
        self.design_page.open(self.sel)
        self.page = "section"

    def open_designer(self, key: str | None = None, airfoil: str | None = None,
                      inc_deg: float | None = None) -> None:
        """The old name for "open the wing": the design page on 3 Wing ▸
        Wing type (gated on a stated mission). `airfoil` and `inc_deg` are
        not taken any more -- the wing's section is chosen on 2 Airfoil from
        AeroBO's library, and its incidence is a row of AeroBO's box."""
        if key is not None:
            self.sel = key
        if not self.mission.stated:
            self.say("state the mission first (step 1 of 2)")
            self.open_mission()
            return
        dp = self.design_page
        if dp.session is None or dp.key != self.sel:
            dp.open(self.sel)
        self.page = "section"
        dp.goto("w.type")

    def goto_view(self, view: str) -> bool:
        """A model's "go there" (aerobo_models.DesignSession.goto)."""
        if view.startswith("m."):
            self.goto_mission(view)
            return True
        if self.page != "section":
            if not self.mission.stated:
                return False
            self.page = "section"
        return self.design_page.goto(view)

    def on_wing_committed(self, key: str) -> None:
        """A design was put on the car: the design page keeps it as its own,
        and the mission's lap (the car as it stands) is stale."""
        self.design_page.on_commit(key)
        self.mission_page.updated_for = None
        self.lib_page.refresh()

    def rederive_stale(self) -> list:
        """Re-derive every fitted AeroBO wing whose slot moved on the car page
        since its law was derived (PLAN2 §5.7) -- synchronously, a few hundred
        milliseconds of AeroBO's evaluator each -- before the build leaves the
        garage (drive, a saved build). Returns the slots re-derived."""
        out = []
        for key in SLOTS:
            spec = self.lib.wings.get(self.build.slot(key).wing)
            if spec is None or spec.engine != "aerobo":
                continue
            am = _am()
            if not am.law_stale(spec, self.build.slot(key)):
                continue
            try:
                if am.rederive(spec, self.build.slot(key), car=self.car):
                    self.lib.save_wing(spec)
                    out.append(key)
                    self.log(f"{wing_name(key)}: WingLab law re-derived at x "
                             f"{self.build.slot(key).x:+.2f} m, h {self.build.slot(key).h:.2f} m",
                             "ok")
            except Exception as exc:                    # noqa: BLE001 -- the old law stands
                self.log(f"{wing_name(key)}: the law could not be re-derived ({exc}); the "
                         f"last one stands", "warn")
        return out

    def open_airfoils(self) -> None:
        """The AIRFOIL LIBRARY: browse and rank carsim's section library (the
        shipped sections, and every section a committed AeroBO wing flies,
        imported with its origin). Reachable from anywhere and not gated:
        reading the library is not designing a wing. Opened against the
        wing in the selected slot, or the role's defaults for an empty one."""
        self._af_return = "section" if self.page == "section" else "car"
        spec = self.lib.wings.get(self.build.slot(self.sel).wing)
        re = spec.reynolds() if spec is not None else 5e5
        self.af_page.open(re, cl_design=1.0, keep=spec.airfoil if spec is not None else None)
        self.page = "airfoil"

    def open_library(self) -> None:
        self.lib_page.refresh()
        self.page = "library"

    def open_saved(self, which: str = "wings") -> None:
        """Task 46: SAVED WINGS ('wings': L, the menu's Saved wings) or SAVED
        CARS ('cars': G, the menu's Saved cars), the cursor on the wing in the
        selected slot / the build in hand when they are in the list. ESC
        comes back to the car page, TAB goes across to the other."""
        if which == "cars":
            self.cars_page.refresh(keep=self.saved_as())
        else:
            self.wings_page.refresh(keep=self.build.slot(self.sel).wing or None)
        self.page = "cars" if which == "cars" else "wings"

    def _saved_changed(self, name: str | None = None) -> None:
        """A wing or build was saved, renamed or deleted (task 46): the card
        pages list again, and SAVED CARS' picture of `name` (every one when
        None: a wing changed, and any build may carry it) is drawn afresh.
        The text library refreshes itself where it always did."""
        self.wings_page.refresh()
        self.cars_page.refresh()
        self.cars_page.invalidate(name)

    def close_page(self) -> None:
        """ESC steps BACK through the design pages rather than dropping
        straight to the car: the chain is the procedure, and walking out of
        the middle of it is how someone ends up with a wing designed against a
        mission they never looked at.

        A FITTED wing (`WingModel.dirty`: fitted from a run, not yet in the
        slot) is saved on the way out -- leaving the design page, and leaving
        the mission page for the car too. Not from the mission page when the
        slot has since been given another wing on the car or library page:
        that design was for the car that was there."""
        dp = self.design_page
        w = dp.wing
        if w is not None and w.dirty and w.spec is not None and (
                self.page == "section"
                or (self.page == "mission" and not dp.car_moved())):
            w.commit()                              # it logs its own "saved ..." line
        if self.page == "airfoil":
            self.page = self._af_return
            return
        back = {"section": "mission", "mission": "car"}
        self.page = back.get(self.page, "car")
        if self.page == "mission":
            self.mission_page.update()

    #: task 45: what ENTER on the airfoil library says when no wing is being
    #: designed (the page opened from the car page) -- keyboard, then pad
    AF_NO_WING = {False: "to use a section, design a wing first: ESC, then D on the car page",
                  True: "to use a section, design a wing first: CIRCLE, then L3 on the car page"}

    def _af_for_wing(self) -> bool:
        """The airfoil page was opened from the design page's section step,
        so ENTER puts the section on the wing being designed."""
        return self._af_return == "section" and self.design_page.af is not None

    def assign_airfoil(self) -> None:
        """ENTER on the airfoil page. Opened from the design page, the
        section is taken as 2 Airfoil's -- IF AeroBO's own library has it
        (the design page flies AeroBO's sections, at AeroBO's library point);
        a section AeroBO does not know is said so, and nothing moves. From
        the car page (A) there is no wing to put it on: it says what to do
        and stays (task 45)."""
        if not self._af_for_wing():
            self.hint = self.AF_NO_WING[self.pad is not None]
            return
        name = self.af_page.current_name()
        if not name:
            return
        af_ = self.design_page.af
        if af_.use_library(name):
            self.page = "section"
            self.design_page.goto("af.section")
        else:
            self.say(af_.msg, "info")

    def prompt_rename(self) -> None:
        """N: name the design before it goes on the car (fitting it first
        when it has not been)."""
        w = self.design_page.wing
        if w is None or (w.spec is None and w.record is None):
            self.say("no wing yet: run the wing first (3 Wing ▸ Solver)", "info")
            return
        if w.spec is None:
            w.fit()
        self._prompt_kind = "rename"
        self.prompt.show("wing name", w.spec.name)

    def prompt_naca(self) -> None:
        self._prompt_kind = "naca"
        self.prompt.show("new NACA 4-digit section (e.g. 4415)", "")

    #: a name prompt a pad can answer too (task 41: `_poll_pad`)
    PROMPT_HINT = "ENTER / CROSS ok   ESC / CIRCLE cancel"

    def prompt_build(self, as_new: bool = False) -> None:
        """The name prompt for saving the car as a build: on its own name
        (ENTER overwrites that build), or -- `as_new`, SHIFT+S / the menu's
        "Save build as" -- on the first free name after it, so ENTER alone
        (or a pad's CROSS) saves a new build beside it."""
        self._prompt_kind = "build"
        base = self.build.name.strip() or default_build_name(self.car)
        self.prompt.show("save the car as a NEW build" if as_new else "save the current car as a build",
                         self.lib.unique_name("builds", base) if as_new else base,
                         hint=self.PROMPT_HINT)

    def prompt_wing(self, as_new: bool = False) -> None:
        """K / SHIFT+K on the car page, the menu's 'Save the side wing as ...'
        (task 46): the name prompt for saving the SELECTED slot's wing as a
        wing of the player's own -- the owner's "save the wings (same as
        saving cars) to test them on different cars". It opens on the wing's
        own name for a wing of the player's (ENTER keeps it: saved in place,
        the car it is on recorded when it had none) and on a FREE name for a
        built-in or with `as_new` (SHIFT+K), so ENTER alone -- a pad's CROSS
        -- saves a new wing beside it. A slot with no wing is refused, and so
        are the study's two published panels ('fin', 'plate'): a copy of one
        under another name would no longer be the published closed form
        (`CarBuild.cfg_kwargs` knows them by name) -- they are on every car
        already."""
        key = self.sel
        what = "top wing" if key == "top" else "side wing"
        w = self.lib.wings.get(self.build.slot(key).wing)
        if w is None:
            self.hint = f"no {what} in the {key} slot to save: W puts one in, D designs one"
            return
        shown = wing_shown(w.name, self.lib)[0]
        if w.legacy:
            self.hint = (f"the {shown} is built in (the study's published panel): it is on "
                         "every car already - nothing to save")
            return
        base = f"my {shown.lower()}" if w.builtin else w.name
        self._prompt_kind = "wing"
        self._wing_from = (key, w.name)
        free = as_new or w.builtin
        if free:
            #  the first free name after the wing's own STEM: a second copy of
            #  'my side-2' is offered 'my side-3', not 'my side-2-2'
            import re
            base = self.lib.unique_name("wings", re.sub(r"-\d+$", "", base) or base)
        self.prompt.show((f"save the {what} as a NEW wing" if free else
                          f"save the {what} as (ENTER keeps its name)"),
                         base, hint=self.PROMPT_HINT)

    def _save_wing(self, name: str) -> bool:
        """The wing prompt's ENTER (task 46): the slot's wing saved as the
        user wing `name` -- a copy of its WingSpec (its design and its aero
        law kept, `builtin` off) recording the car it is on now
        (`WingSpec.made_for`). The wing's OWN name saves it in place (only
        the car recorded, when it had none); a name another wing's file
        holds is never written over (saved beside it, '-2', as a build's
        name is) -- a wing is named by the builds that carry it. The slot
        (both sides, mirrored) then carries the saved wing, as S makes the
        car in hand the saved build. False, with the hint, when nothing
        was saved."""
        from .prerace import car_label
        key, src_name = getattr(self, "_wing_from", (self.sel, ""))
        src = self.lib.wings.get(src_name)
        what = "top wing" if key == "top" else "side wing"
        if src is None:
            self.hint = f"'{src_name}' is no longer in the library"
            return False
        name = str(name or "").strip()[:32]
        if not name:
            return False
        if name == src.name and not src.builtin:
            if src.made_for:
                self.hint = (f"wing '{name}' is saved (made on the {car_label(src.made_for)}; "
                             "nothing changed)")
                return True
            src.made_for = self.car
            try:
                self.lib.save_wing(src)
            except (OSError, ValueError) as exc:
                src.made_for = ""
                self.hint = _could_not_save(exc)
                return False
            self._saved_changed()
            self.hint = f"wing '{name}' saved (made on the {car_label(self.car)})"
            return True
        new = self.lib.unique_name("wings", name)
        copy = src.copy(name=new, builtin=False, made_for=self.car)
        try:
            self.lib.save_wing(copy)
        except (OSError, ValueError) as exc:
            self.hint = _could_not_save(exc)
            return False
        slot = self.build.slot(key)
        if slot.wing == src.name:
            slot.wing = new
            self.build.sync_mirror(key)
            self.build.clamp(self.lib, self.car)
        self._saved_changed()
        self.hint = ((f"'{name}' is taken - " if new != name else "")
                     + f"the {what} saved as '{new}' (made on the {car_label(self.car)}), "
                       f"now in the {key} slot; L lists it")
        return True

    def prompt_rename_build(self, name: str | None = None) -> None:
        """R on the library's BUILDS list (task 41) -- or on a SAVED CARS
        card, `name` (task 46): rename that build (`Library.rename`)."""
        if name is None:
            it = self.lib_page.builds.current()
            name = it[0] if it else None
        if not name or name not in self.lib.builds:
            self.hint = "no build here to rename"
            return
        if self.lib.builds[name].get("builtin"):
            self.hint = "built-in builds cannot be renamed"
            return
        self._prompt_kind = "rename_build"
        self._rename_from = name
        self.prompt.show(f"rename the build '{name}'", name, hint=self.PROMPT_HINT)

    def _prompt_done(self, value: str) -> None:
        kind = self._prompt_kind
        value = value.strip()
        if not value:
            return
        w = self.design_page.wing
        if kind == "rename" and w is not None and w.spec is not None:
            w.spec.name = value[:32]
            w.dirty = True
        elif kind == "naca":
            code = "".join(ch for ch in value if ch.isdigit())[:4]
            if len(code) != 4:
                self.hint = "a NACA 4-digit code is four digits"
                return
            from .aero.airfoil import AirfoilSpec
            name = f"naca{code}"
            if name not in self.lib.airfoils:
                try:
                    self.lib.save_airfoil(AirfoilSpec(name, "naca", code=code, notes="user NACA 4-digit"))
                except (OSError, ValueError) as exc:
                    self.hint = _could_not_save(exc)
                    return
            self.af_page.open(self.af_page._re, keep=name)
            self.hint = f"section {name} added"
        elif kind == "build":
            #  the SAME name overwrites that build (it was asked for); a name
            #  that only folds onto another build's file ('Kestrel Fast' /
            #  'kestrel-fast') is saved beside it instead -- and so (task 41)
            #  is a name another CAR's build holds: typing 'bus wings' on the
            #  Corsa must not re-tag the bus's build as a Corsa one
            from .prerace import car_label
            name = value[:32]
            held = self.lib.builds.get(name)
            if held is not None and not self._own_build(name):
                new = self.lib.unique_name("builds", name)
                c = held.get("car", "") if isinstance(held, dict) else ""
                other = self._default_of_other(name)
                why = (f"'{name}' is the {car_label(c)}'s build" if c else
                       f"'{name}' is the {car_label(other)}'s default" if other else
                       f"'{name}' is built in")
            else:
                new = name if held is not None else self.lib.unique_name("builds", name)
                why = f"'{name}' already exists"
            if not self._write_build(new):
                return
            self.hint = (f"build '{new}' saved to the library" if new == name
                         else f"{why} - saved as '{new}'")
        elif kind == "rename_build":
            self._rename_build(getattr(self, "_rename_from", ""), value)
        elif kind == "wing":                         # task 46: K / SHIFT+K
            self._save_wing(value)

    # -- slot editing (car page) ---------------------------------------------
    def _past_limit(self, key: str, spec: "WingSpec", h: float | None = None) -> float | None:
        """This slot's physical span limit on this car when `spec` would be
        past it (mounted at `h`, default the slot's own), else None
        (`bodies.span_limit`, with `over_limits`' tolerance)."""
        h = self.build.slot(key).h if h is None else h
        lim = bodies.span_limit(SLOT_ROLE[key], self.car, h)
        return lim if float(spec.span) > lim + bodies.LIMIT_TOL else None

    def _limits_past(self) -> dict:
        """{slot: its limit} for every fitted wing past its physical limit on
        this car now (`limit_rows`: the SPAN LIMITS panel's own test, so the
        hint and the panel cannot disagree)."""
        return {k: lim for k, _span, lim, past in limit_rows(self.build, self.lib, self.car)
                if past}

    def _limit_cross(self, was: dict, keys=None) -> tuple | None:
        """Task 45: the first slot of `keys` (default the selected one) that
        an edit took across its span limit on this car, from `was` (the
        `_limits_past` before it), as (slot, what it means): 'now past the
        Corsa's 0.50 m limit here: runs count as UNLIMITED (not official)',
        or 'within the Corsa's limit again'. None when nothing crossed -- so
        it is said ONCE, at the crossing. A player in Unlimited mode lowering
        a flank used to make the build unofficial with only the SPAN LIMITS
        panel going red."""
        from .prerace import car_label
        now = self._limits_past()
        who = car_label(self.car) or "car"
        for key in keys or (self.sel,):
            if (key in now) == (key in was):
                continue
            if key in now:
                here = "" if key == "top" else " here"       # a flank's limit is its height's
                return key, (f"now past the {who}'s {now[key]:.2f} m limit{here}: "
                             "runs count as UNLIMITED (not official)")
            still = [k for k in SLOTS if k in now]
            return key, (f"within the {who}'s limit again" + (
                f" ({' and '.join(still)} still past it: runs stay UNLIMITED)" if still else ""))
        return None

    def _cycle_wing(self, d: int = 1) -> None:
        """W: the next library wing of the slot's role. In Real mode (task 41)
        a wing past this slot's span limit on this car is SKIPPED, and the
        hint says why and where that changes. A wing that takes the slot past
        its limit (Unlimited) or back says what that does to runs (task 45)."""
        was = self._limits_past()
        slot = self.build.slot(self.sel)
        role = SLOT_ROLE[self.sel]
        names = [""] + sorted(n for n, w in self.lib.wings.items() if w.role == role)
        i = names.index(slot.wing) if slot.wing in names else 0
        skipped, lim = [], None
        for _ in range(len(names)):
            i = (i + d) % len(names)
            w = self.lib.wings.get(names[i]) if names[i] else None
            past = None if (w is None or self.unlimited) else self._past_limit(self.sel, w)
            if past is None:
                break
            skipped.append(names[i])
            lim = past
        slot.wing = names[i]
        self.build.sync_mirror(self.sel)
        self.build.clamp(self.lib, self.car)
        #  task 45: the wing by the name a player reads, where it is in the
        #  cycle and what it does -- 'left: Side plate (3 of 4) - end plates:
        #  ...'. 'no wing' is counted as the cycle's LAST step (it is where
        #  one more W goes after the last wing), so it reads '(4 of 4)'.
        shown, what = wing_shown(slot.wing, self.lib)
        self.hint = (f"{self.sel}: {shown} ({i or len(names)} of {len(names)})"
                     + (f" - {what}" if what else ""))
        if skipped:
            self.hint += (f"  ({len(skipped)} skipped: past this slot's {lim:.2f} m span limit;"
                          f" Settings > Wing limits: Unlimited allows them)")
        cross = self._limit_cross(was)
        if cross is not None:
            self.hint += f", {cross[1]}"

    def try_ready_made(self) -> bool:
        """The TIME TRIAL page's 'Try ready-made wings' (round 3 of task 45,
        the owner: wings sooner): the garage opens on the car page with the
        first EMPTY slot selected (left, right, top) and W pressed once on
        it -- a ready-made wing on the car in one key -- and the hint says
        what the next presses do. False, and nothing changed, when every
        slot has a wing already (the page offers the row only to a car with
        none)."""
        key = next((k for k in SLOTS if not self.build.slot(k).wing), None)
        if key is None:
            return False
        self.page = "car"
        self._select(key)
        self._cycle_wing(1)
        if self.build.slot(key).wing:
            self.hint += "  -  W: the next one, ENTER: drive it"
            self._hint_guide = self.hint
        return True

    def _bg_hint(self, msg: str) -> None:
        """A note from the background -- the wing data arriving, on a first
        visit while the player looks at a car -- is the hint, unless the
        guide hint of 'Try ready-made wings' is still up (round 3: the first
        garage of a new player computes wing data at once, and its note
        covered what W and ENTER do next)."""
        if self._hint_guide and self.hint == self._hint_guide and self.hint_alpha() > 0.0:
            return
        self.hint = msg

    def _h_stop(self, key: str, h_old: float, span: float | None = None) -> bool:
        """The stop on a FLANK slot moved down. Real mode (task 41): the
        fitted panel's lower tip may come to the car's ground clearance and
        no further, h >= ground + span / 2 (`flank_h_floor`). Unlimited mode
        (task 45): past the clearance, but not into the road -- the lower tip
        stops at 0 m, h >= span / 2. A slot already lower than its stop (a
        build loaded, or saved, past it) is left where it is and does not
        move further down. `span` is the panel's (default the slot's library
        wing). True when it stopped the move; the hint says so."""
        if key == "top":
            return False
        slot = self.build.slot(key)
        if span is None:
            w = self.lib.wings.get(slot.wing) if slot.wing else None
            if w is None:
                return False
            span = w.span
        floor = flank_h_floor(self.car, span, self.unlimited)
        #  the band's own floor may hold the slot right AT the stop (a 0.80 m
        #  panel's road stop is the Corsa band's 0.40 m): said all the same
        held = abs(slot.h - h_old) <= 1e-12 and abs(slot.h - floor) <= 1e-9
        if not held:
            if slot.h >= floor - 1e-9 or slot.h >= h_old:
                return False
            slot.h = min(h_old, floor)
        if self.unlimited:
            self.hint = (f"{key}: the panel's lower tip is at the road (h {floor:.2f} m "
                         f"for {span:.2f} m) - it goes no lower")
        else:
            self.hint = (f"{key}: the panel's lower tip is at the ground clearance (h {floor:.2f} m "
                         f"for {span:.2f} m) - Settings > Wing limits: Unlimited goes lower")
        return True

    def _move(self, dx: float = 0.0, dh: float = 0.0, dinc: float = 0.0) -> None:
        was = self._limits_past()
        slot = self.build.slot(self.sel)
        h_old = slot.h
        slot.x += dx
        slot.h += dh
        slot.inc_deg += dinc
        want_x = slot.x
        self.build.clamp(self.lib, self.car)
        if self.sel == "top" and dx < 0.0 and slot.x > want_x + 1e-9:
            #  task 46: held where its pylons / plates still stand on the car
            w = self.lib.wings.get(slot.wing)
            if w is not None and abs(slot.x - top_x_floor(self.car, w)) < 1e-9:
                self.hint = ("top wing: its " + ("pylons" if w.mount == "pylon" else "endplates")
                             + " must stand on the car - it goes no further back")
        if dh < 0.0:
            self._h_stop(self.sel, h_old)
        self.build.sync_mirror(self.sel)
        #  task 45: a flank's limit is its height's, so a move can take it
        #  past the limit (Unlimited) or back: said once, at the crossing --
        #  over the stop's hint when one move does both
        cross = self._limit_cross(was)
        if cross is not None:
            self.hint = f"{cross[0]} is {cross[1]}"

    def _select(self, key: str) -> None:
        self.sel = key

    # -- pad ----------------------------------------------------------------
    def _attach_pad(self) -> None:
        try:
            from .input import GamepadInput
            if GamepadInput.available():
                self.pad = GamepadInput(0, steer_limit=False)
                self._pad_seeded = False
                print(f"garage: gamepad {self.pad.name} ({self.pad.layout} layout)")
        except Exception as exc:
            print(f"garage: gamepad not usable ({exc})")
            self.pad = None

    def _pad_edge(self, name: str) -> bool:
        now = bool(self.pad.pressed(name)) if self.pad is not None else False
        was = self._pad_prev.get(name, False)
        self._pad_prev[name] = now
        return now and not was

    def _poll_pad_menu(self) -> str | None:
        p = self.pad
        edges = {n: self._pad_edge(n) for n in PAD_NAMES_POLLED}
        action = None
        for name, cmd in (("up", "nav_up"), ("down", "nav_down"), ("cross", "select"),
                          ("circle", "back"), ("options", "menu")):
            if edges[name]:
                action = self.menu.handle(cmd) or action
        nav = self._menu_stick.poll(p.stick("left")[1])
        if nav:
            self.menu.handle(nav)
        return self._menu_action(action)

    def _poll_pad(self, dt: float) -> str | None:
        p = self.pad
        if p is None:
            return None
        if not self._pad_seeded:
            self._pad_prev = {n: bool(p.pressed(n)) for n in PAD_NAMES_POLLED}
            self._pad_seeded = True
            return None
        if self.menu.open:
            return self._poll_pad_menu()
        if self.prompt.open:
            #  task 41: a pad answers a name prompt (the menu's "Save build
            #  as", which opens on a free name): CROSS takes the name as it
            #  stands, CIRCLE cancels. Typing stays the keyboard's.
            e = {n: self._pad_edge(n) for n in PAD_NAMES_POLLED}
            if e["cross"]:
                self.prompt.open_ = False
                self._prompt_done(self.prompt.value)
            elif e["circle"]:
                self.prompt.open_ = False
            return None
        e = {n: self._pad_edge(n) for n in PAD_NAMES_POLLED}
        lx, ly = p.stick("left")
        rx, ry = p.stick("right")
        if self.page == "car":
            if abs(lx) > 0.0 or abs(ly) > 0.0:
                self._move(dx=lx * self.PAD_MOVE_X * dt, dh=-ly * self.PAD_MOVE_H * dt)
            if abs(rx) > 0.0 or abs(ry) > 0.0:
                self.cam.orbit(-rx * self.PAD_ORBIT * dt, ry * self.PAD_ORBIT * dt)
            if e["r1"]:
                self._move(dinc=STEP_INC)
            if e["l1"]:
                self._move(dinc=-STEP_INC)
            if e["up"]:
                self._move(dh=STEP_H)
            if e["down"]:
                self._move(dh=-STEP_H)
            if e["right"]:
                self._move(dx=STEP_X)
            if e["left"]:
                self._move(dx=-STEP_X)
            if e["triangle"]:
                self.sel = SLOTS[(SLOTS.index(self.sel) + 1) % 3]
            if e["square"]:
                self._cycle_wing()
            if e["circle"]:
                self.deploy_cmd = 1.0 - self.deploy_cmd
            if e["r3"]:
                self.cam.reset()
            if e["l3"]:
                self.open_mission()
            if e["cross"]:
                return "drive"
            if e["options"]:
                self._menu_open()
            return None
        if self.page in SHELL_PAGES:
            #  the AeroBO shell takes the pad as keys: the left stick is folded
            #  into the d-pad edges here (with the list pages' repeat), OPTIONS
            #  stays the garage's pause menu
            nav = self._nav_stick.poll(ly)
            e["up"] = e["up"] or nav == "nav_up"
            e["down"] = e["down"] or nav == "nav_down"
            self._rep_t -= dt
            if abs(lx) > 0.6 and self._rep_t <= 0.0:
                e["right" if lx > 0 else "left"] = True
                self._rep_t = self.PAD_REPEAT
            if abs(lx) < 0.3:
                self._rep_t = 0.0
            if e["options"]:
                self._menu_open()
                return None
            self.shell.pad(e, lx, ly, bool(p.pressed("l1")))
            return None
        # the list pages: stick / d-pad navigate, L1 = fine, cross = activate,
        # circle = back, square / triangle = page actions
        nav = self._nav_stick.poll(ly)
        if e["up"] or nav == "nav_up":
            self._page_nav(-1)
        if e["down"] or nav == "nav_down":
            self._page_nav(+1)
        fine = bool(p.pressed("l1"))
        self._rep_t -= dt
        if e["right"] or (lx > 0.6 and self._rep_t <= 0.0):
            self._page_adjust(+1, fine)
            self._rep_t = self.PAD_REPEAT
        if e["left"] or (lx < -0.6 and self._rep_t <= 0.0):
            self._page_adjust(-1, fine)
            self._rep_t = self.PAD_REPEAT
        if abs(lx) < 0.3:
            self._rep_t = 0.0
        if e["cross"]:
            self._page_activate()
        if e["circle"]:
            self.close_page()
        if e["square"]:
            if self.page == "airfoil":
                self.af_page.request_xfoil()
            elif self.page in ("library", "cars"):
                self._save_build_quick()
            elif self.page == "wings":                 # task 46: the slot's wing, named
                self.prompt_wing()
        if e["r1"] and self.page == "library":         # task 41: the car's default
            lp = self.lib_page
            if lp.focus == "builds" and lp.builds.current():
                self.make_default(lp.builds.current()[0])
            else:
                self.hint = "R1 makes a BUILD the car's default: TRIANGLE to the builds"
        elif e["r1"] and self.page == "cars":          # task 46: its car's default
            self._make_default_saved(self.cars_page.current())
        if self.page == "wings" and (e["r1"] or e["l1"]):
            #  task 46: the slot the cards are judged for, next / previous
            self._select(SLOTS[(SLOTS.index(self.sel) + (1 if e["r1"] else -1)) % 3])
        if e["triangle"]:
            if self.page == "library":
                self.lib_page.focus = "builds" if self.lib_page.focus == "wings" else "wings"
            elif self.page == "airfoil":
                self.af_page.focus = "params" if self.af_page.focus == "list" else "list"
            elif self.page in ("wings", "cars"):       # task 46: across to the other
                self.open_saved("cars" if self.page == "wings" else "wings")
        if e["options"]:
            self._menu_open()
        return None

    # -- page-generic navigation ------------------------------------------------
    #  (the mission and design pages are the AeroBO shell's: `self.shell`)
    def _page_nav(self, d: int) -> None:
        if self.page == "airfoil":
            if self.af_page.focus == "list":
                self.af_page.list.nav(d)
            else:
                self.af_page.params.nav(d)
        elif self.page == "library":
            (self.lib_page.wings if self.lib_page.focus == "wings" else self.lib_page.builds).nav(d)
        elif self.page in ("wings", "cars"):          # task 46: UP / DOWN a row of cards
            (self.wings_page if self.page == "wings" else self.cars_page).nav_row(d)

    def _page_adjust(self, d: int, fine: bool) -> None:
        if self.page == "airfoil" and self.af_page.focus == "params":
            self.af_page.params.adjust(d, fine)
        elif self.page == "airfoil":
            self.af_page.list.nav(d)
        elif self.page in ("wings", "cars"):          # task 46: LEFT / RIGHT a card
            (self.wings_page if self.page == "wings" else self.cars_page).nav(d)

    def _page_activate(self) -> None:
        #  (ENTER on the mission page states the mission through the shell:
        #  `Garage.state_mission`, the one entry point, design_shell ENTER)
        if self.page == "airfoil":
            p = self.af_page.params.current()
            if self.af_page.focus == "list" or (p is not None and p.key == "use"):
                #  the 'use' row goes to assign_airfoil even dimmed: with no
                #  wing being designed it cannot fire, and that says why
                self.assign_airfoil()
            else:
                self.af_page.params.activate()
        elif self.page == "library":
            self._library_select()
        elif self.page == "wings":                    # task 46
            self._fit_saved_wing(self.wings_page.current())
        elif self.page == "cars":
            self._saved_car_select(self.cars_page.current())

    # -- THE MOUSE ---------------------------------------------------------------
    #  Every page is driven by the keyboard OR the mouse, and the mouse does
    #  the SAME things: it lands on the row the drawing put under the cursor
    #  and does there what the keyboard would do on that row. The navigator's
    #  gate applies to a click exactly as it applies to an arrow key -- the
    #  UROP app's own shell refuses a locked node's click and notifies the
    #  reason, which is what the design shell's `select` does on its pages.
    def _page_widgets(self):
        """(widget, focus-name) pairs a click may land on, on THIS page (the
        airfoil and library pages; the shell takes its own clicks)."""
        if self.page == "airfoil":
            return [(self.af_page.list, "list"), (self.af_page.params, "params")]
        if self.page == "library":
            return [(self.lib_page.wings, "wings"), (self.lib_page.builds, "builds")]
        if self.page == "wings":
            return [(self.wings_page, "wings")]
        if self.page == "cars":
            return [(self.cars_page, "cars")]
        return []

    def _page_click(self, pos, fine: bool) -> None:
        if self.page == "wings" and self.wings_page.chip_hit(pos):
            self._select(self.wings_page.chip_hit(pos))     # task 46: a slot chip
            return
        for w, focus in self._page_widgets():
            if w.hit(pos) < 0:
                continue
            self._set_focus(focus)
            if isinstance(w, (ui.ListBox, _CardGrid)):
                if w.click(pos) == "activate":
                    self._page_activate()
                return
            r = w.click(pos, fine)
            if r == "action" and self.page == "airfoil" and not self._af_for_wing() \
                    and w.current().key == "use":
                self.assign_airfoil()            # the dimmed 'use' row: say why (task 45)
            if r in ("action", "adjust"):
                self._after_rows_changed()
            return

    def _page_wheel(self, dy: int) -> None:
        if self.page in ("wings", "cars"):
            #  task 46: the wheel scrolls the cards wherever it is (a gap
            #  between two cards, the header)
            (self.wings_page if self.page == "wings" else self.cars_page).wheel(dy)
            return
        pos = pygame.mouse.get_pos()
        for w, focus in self._page_widgets():
            if w.hit(pos) < 0:
                continue
            self._set_focus(focus)
            w.wheel(dy)
            return

    def _set_focus(self, focus: str) -> None:
        if self.page == "airfoil":
            self.af_page.focus = focus
        elif self.page == "library":
            self.lib_page.focus = focus

    def _after_rows_changed(self) -> None:
        """A row fired or stepped: the gates may have moved with it. On the
        shell pages the page moves only if its whole STAGE locked (a shut
        view inside an open stage stays: AeroBO shows its empty state), and
        nothing is echoed -- the models log what they did, refusals toast."""
        if self.page in SHELL_PAGES:
            self.shell.settle()

    def _run_optimiser(self, extend: bool = False) -> None:
        """O / K / SQUARE and ENTER on the wing's Solver / Convergence: start
        the current group's search LIVE (`DesignPage.optimise`). Nothing
        blocks and nothing is pre-drawn: the frame pumps the run."""
        self.design_page.optimise(extend=extend)

    def _library_select(self) -> None:
        lp = self.lib_page
        if lp.focus == "wings":
            it = lp.wings.current()
            if not it:
                return
            w = self.lib.wings.get(it[0])
            if w is None:
                return
            if w.role != SLOT_ROLE[self.sel]:
                self.hint = f"'{w.name}' is a {w.role} wing: select a matching slot first"
                return
            #  Real mode (task 41): the library page fits no wing past this
            #  slot's span limit on this car either -- the same rule as W
            lim = None if self.unlimited else self._past_limit(self.sel, w)
            if lim is not None:
                self.hint = (f"'{w.name}' ({w.span:.2f} m) is past this slot's {lim:.2f} m "
                             f"span limit; Settings > Wing limits: Unlimited allows it")
                return
            was = self._limits_past()
            slot = self.build.slot(self.sel)
            slot.wing = w.name
            self.build.sync_mirror(self.sel)
            self.build.clamp(self.lib, self.car)
            cross = self._limit_cross(was)       # as W says it (task 45)
            self.hint = f"{self.sel}: {w.name}" + (f", {cross[1]}" if cross else "")
            self.page = "car"
        else:
            it = lp.builds.current()
            if not it:
                return
            self._load_build(it[0])
            self.page = "car"

    # -- the SAVED WINGS / SAVED CARS pages (task 46) --------------------------
    def _fit_saved_wing(self, name: str | None) -> bool:
        """ENTER on a SAVED WINGS card: the wing into the selected slot (both
        sides, mirrored) when it FITS (`wing_fit`: the slot's role, this car's
        span limit at the slot, x3 in Unlimited), and back to the car page to
        see it on the car. One that does not fit stays out, and the hint says
        why and what would make it fit: the slot a top / side wing wants, the
        height a side wing fits from on this car (its limit grows with the
        height), Settings > Wing limits: Unlimited."""
        w = self.lib.wings.get(name) if name else None
        if w is None:
            self.hint = "no wing here"
            return False
        shown = wing_shown(w.name, self.lib)[0]
        fits, why, _lim, _past = wing_fit(w, self.sel, self.build, self.car, self.unlimited)
        if not fits:
            more = ""
            if w.role == SLOT_ROLE[self.sel]:
                h = fit_height(w, self.car, self.unlimited)
                if h is not None:
                    more = f" - it fits from h {h:.2f} m (UP on the car page)"
                if not self.unlimited:
                    more += ("; " if more else " - ") + "Settings > Wing limits: Unlimited allows it"
            else:
                more = " (1 / 2 / 3, or R1)" if self.pad is not None else " (1 / 2 / 3)"
            self.hint = f"'{shown}' is {why}{more}" if why.startswith("a ") else f"'{shown}': {why}{more}"
            return False
        was = self._limits_past()
        slot = self.build.slot(self.sel)
        slot.wing = w.name
        self.build.sync_mirror(self.sel)
        self.build.clamp(self.lib, self.car)
        cross = self._limit_cross(was)                  # as W says it (task 45)
        self.hint = f"{self.sel}: {shown}" + (f", {cross[1]}" if cross else "")
        self.page = "car"
        return True

    def _saved_car_select(self, name: str | None) -> str:
        """ENTER on a SAVED CARS card (task 46). A build of THIS car -- or an
        any-car one -- is loaded (`_load_build`, as the library's BUILDS list
        loads it) and the car page shows it: 'load'. Another car's build, in
        the drive's garage (Settings, not a challenge's car): the garage ends
        with 'car' (`_pending_action`), `car_wanted` its car and
        `build_wanted` the build, and the drive opens the garage on that car
        with that build (drive._garage_car): 'car'. In a bare garage, or on a
        challenge's car, it only says whose it is: ''."""
        from .prerace import car_label
        if not name or name not in self.lib.builds:
            self.hint = "no build here"
            return ""
        c = self.cars_page.car_of(name)
        if not c or c == self.car:
            self._load_build(name)
            self.page = "car"
            return "load"
        if self.settings is None or self.car_fixed:
            self.hint = (f"'{name}' is the {car_label(c)}'s build: "
                         + (f"this challenge drives the {car_label(self.car)}" if self.car_fixed
                            else "change car from the drive's garage to load it"))
            return ""
        self._leave_runs()
        self.car_wanted, self.build_wanted = c, name
        self.hint = f"to the {car_label(c)}, with '{name}'"
        self._pending_action = "car"
        return "car"

    def _make_default_saved(self, name: str | None) -> None:
        """D (pad R1) on a SAVED CARS card (task 46): the build becomes ITS
        car's default -- this car's for one of this car or an any-car one
        (`make_default`), else that other car's (Settings.car_build[its
        car]), as the drive's Settings > Default would make it there."""
        from .prerace import car_label
        if not name or name not in self.lib.builds:
            self.hint = "no build here"
            return
        c = self.cars_page.car_of(name)
        if not c or c == self.car:
            self.make_default(name)
            return
        if self.settings is None:
            self.hint = "no drive settings here: set a car's default from the drive's garage"
            return
        if self._set_default(name, car=c):
            self.hint = f"'{name}' is the {car_label(c)}'s default build"
            self.lib_page.refresh()

    def _delete_saved(self, kind: str, name: str | None) -> bool:
        """DEL on a library item (the text library's lists, task 45; the card
        pages, task 46): `kind` 'wings' / 'builds', the item `name`. The
        first DEL only arms it and says what goes (and when it is a car's
        default, or on the car now); a second DEL on the same item, within
        ARM_S, deletes it. True when it went."""
        from .prerace import car_label
        if kind == "wings":
            if not name or name not in self.lib.wings:
                return False
            if self.lib.wings[name].builtin:
                self.hint = "built-in wings cannot be deleted"
                return False
            fitted = any(self.build.slot(k).wing == name for k in SLOTS)
            if not self._confirm(f"del:wings:{name}",
                                 f"DEL again: delete wing {name}"
                                 + (" (on the car now: it comes off)" if fitted else ""),
                                 (pygame.K_DELETE,)):
                return False
            self.lib.delete("wings", name)
            self.build.clamp(self.lib, self.car)
            self.hint = f"deleted wing '{name}'"
            self._saved_changed()
            return True
        if not name or name not in self.lib.builds:
            return False
        cb = getattr(self.settings, "car_build", None)
        of = [c for c, n in cb.items() if n == name] if isinstance(cb, dict) else []
        if not self._confirm(f"del:builds:{name}",
                             f"DEL again: delete build {name}"
                             + (f" (the {', '.join(car_label(c) for c in of)}'s default)"
                                if of else ""),
                             (pygame.K_DELETE,)):
            return False
        self.lib.delete("builds", name)
        self.hint = f"deleted build '{name}'"
        #  a car whose default it was has none now (task 41): said here,
        #  rather than found out the next time that car is chosen
        cb = getattr(self.settings, "car_build", None)
        gone = [c for c, n in cb.items() if n == name] if isinstance(cb, dict) else []
        if gone:
            self.settings.car_build = {c: n for c, n in cb.items() if n != name}
            self.settings.save()
            self.hint += (f" - it was the {', '.join(car_label(c) for c in gone)}'s "
                          f"default: none now")
        self._saved_changed(name)
        return True

    # -- builds: save, switch, the car's own default (task 41) ---------------
    #  The owner: "There has to be an easy way to save and change the wing
    #  cars. Also a custom default for each car the user wants." On the car
    #  page S saves (in place when the car already IS one of this car's saved
    #  builds, else it asks the name), SHIFT+S saves as a new name, B / SHIFT+B
    #  steps through this car's saved builds the way W steps a slot's wings,
    #  and F makes the car in hand this car's DEFAULT: the build the drive
    #  loads when the player switches to this car (drive.drive, Settings.
    #  car_build). A pad has the same four in the pause menu (OPTIONS), whose
    #  buttons on the car page are all taken; the library page adds D (R1),
    #  R and DEL on the build under the cursor.
    def _fit_in(self, src: CarBuild) -> CarBuild:
        """The build the garage edits, from `src` (review of task 41, root
        design). THIS car's own build is edited in place, clamped to this
        car's bands. Any other -- an any-car build from before task 41,
        another car's -- is only being LOOKED at on this car: the garage
        edits a fitted COPY and keeps `src`, so merely opening it here (or
        loading it from the library) and leaving moves nothing in it;
        `handed_back` returns `src` untouched unless the copy was changed.
        Task 45: a build still called by another car's new-build name ('my
        corsa' on the Express, `other_car_name`) is called this car's ('my
        express') -- in place when it is this car's own, on the copy
        otherwise (looked at only, `src` goes back with its name)."""
        rename = other_car_name(src, self.car, self.lib)
        if getattr(src, "car", "") == self.car:
            self._held = None
            if rename:
                src.name = default_build_name(self.car)
            return src.clamp(self.lib, self.car)
        fitted = CarBuild.from_json(src.to_json()).clamp(self.lib, self.car)
        if rename:
            fitted.name = default_build_name(self.car)
        self._held = (src, fitted.to_json())
        return fitted

    def handed_back(self) -> CarBuild:
        """The build the garage gives the drive when it closes: `self.build`,
        except an any-car / other-car build that was only looked at here --
        then the build as it came in, not its copy fitted to this car (a
        pre-41 build viewed on the bus keeps its Corsa-band stations; the
        drive fits a copy to whatever car it is driven on)."""
        if self._held is not None and self.build.to_json() == self._held[1]:
            return self._held[0]
        return self.build

    def default_name(self) -> str:
        """This car's default build (the drive's Settings.car_build), by
        library name; "" when none was chosen or the garage has no Settings."""
        bo = getattr(self.settings, "build_of", None)
        return bo(self.car) if bo is not None else ""

    def _own_build(self, name: str) -> bool:
        """Is `name` a user build in the library that this car may write
        over: its own, or an any-car one from before task 41? Another car's
        build (or a built-in) is saved BESIDE, never over -- and so is an
        any-car build that ANOTHER car's Settings default names (review of
        task 41, finding 12): writing over it, and stamping this car on it,
        would edit that car's default from here and drop it from its B list."""
        b = self.lib.builds.get(name)
        if not (isinstance(b, dict) and not b.get("builtin")
                and b.get("car", "") in ("", self.car)):
            return False
        return not (b.get("car", "") == "" and self._default_of_other(name))

    def _default_of_other(self, name: str) -> str:
        """The first OTHER car whose Settings default is the build `name`
        ("" when none, or the garage has no Settings)."""
        cb = getattr(self.settings, "car_build", None)
        if not isinstance(cb, dict):
            return ""
        return next((c for c, n in sorted(cb.items()) if n == name and c != self.car), "")

    def saved_as(self) -> str | None:
        """The library build the car in hand IS -- same wings, stations and
        angles (`prerace._same_build`) -- its own name first; None: unsaved."""
        from .prerace import _same_build
        js = self.build.to_json()
        if _same_build(self.lib.builds.get(self.build.name), js):
            return self.build.name
        for n, b in self.lib.builds.items():
            if _same_build(b, js):
                return n
        return None

    def build_header(self) -> list:
        """The car page's BUILD line (task 45: the page never said which
        build was in hand, or that an edit to it was not saved), as the
        [(text, colour)] `GarageView` draws:
            BUILD  Fast Wings   saved  .  Corsa default         (a middle dot)
            BUILD  Fast Wings   unsaved changes (S saves)
            BUILD  my corsa   not saved yet (S saves it)    (a name the library lacks)
            BUILD  (unnamed)   S saves it
        'saved' is `saved_as` (the car in hand IS that library build; a
        built-in one says 'built-in'), the default `default_name`."""
        from .prerace import car_label
        name = self.build.name.strip()
        segs = [("BUILD  ", C_TEXT_DIM), (name or "(unnamed)", C_TEXT if name else C_TEXT_DIM)]
        if not name:
            return segs + [("   S saves it", C_PANEL_ON)]
        saved = self.saved_as()
        if saved is not None:
            state = "built-in" if (self.lib.builds.get(saved) or {}).get("builtin") else "saved"
            segs.append((f"   {state}" if saved == name else f"   {state} as '{saved}'", C_OK))
            if saved == self.default_name():
                segs.append((f"  \u00b7  {car_label(self.car)} default", C_TEXT_DIM))
        elif name in self.lib.builds:
            segs.append(("   unsaved changes (S saves)", C_PANEL_ON))
        elif not self.build.has_any(self.lib):
            segs.append(("   no wings yet", C_TEXT_DIM))
        else:
            segs.append(("   not saved yet (S saves it)", C_PANEL_ON))
        return segs

    def car_builds(self) -> list:
        """The builds B steps through: this car's own, then the any-car ones
        (`prerace.pick_order`). Other cars' builds are the library page's."""
        from .prerace import pick_order
        return [n for n in pick_order(self.lib.builds, self.car)
                if self.lib.builds[n].get("car", "") in ("", self.car)]

    def _write_build(self, name: str) -> bool:
        """Save the car in hand as the library build `name`, stamped with this
        car. False (and the hint says why) when the write failed; the car
        keeps its old name and tag then."""
        self.rederive_stale()              # a saved build carries current laws
        was = self.build.name, self.build.car
        self.build.name, self.build.car = name, self.car
        try:
            self.lib.save_build(self.build.to_json())
        except (OSError, ValueError) as exc:
            self.build.name, self.build.car = was
            self.hint = _could_not_save(exc)
            return False
        self.lib_page.refresh()
        self._saved_changed(name)            # task 46: its picture is drawn afresh
        return True

    def save_build(self, as_new: bool = False) -> None:
        """S: the car in hand into the library. Already one of this car's
        saved builds under its name: written over in place (nothing changed:
        said, not rewritten). Anything else -- a new car, another car's build
        -- asks the name first. SHIFT+S (`as_new`): the prompt, on a free name."""
        from .prerace import _same_build
        n = self.build.name
        if as_new or not self._own_build(n):
            self.prompt_build(as_new=as_new)
            return
        held = self.lib.builds[n]
        if _same_build(held, self.build.to_json()) and held.get("car", "") == self.car:
            self.hint = f"build '{n}' is saved (nothing changed)"
            return
        if self._write_build(n):
            self.hint = f"build '{n}' saved"

    def _save_build_quick(self) -> None:
        """The pad's SQUARE on the library page: saved WITHOUT a prompt. The
        car's own saved build is written over in place (it used to pile up
        'my corsa-2', '-3' ... on every press); anything else is saved under
        the first free name after its own. The menu's "Save build as" is the
        explicit new-name path."""
        n = self.build.name
        name = n if self._own_build(n) else self.lib.unique_name("builds", n)
        if self._write_build(name):
            self.hint = f"build '{name}' saved"

    def _load_build(self, name: str) -> str:
        """The library build `name` becomes the car in hand (fitted to this
        car; another car's or an any-car one as a copy, `_fit_in`). Task 45:
        a load that takes a slot past its span limit on this car, or back,
        says so -- the selected slot first -- and returns that tail ("" when
        nothing crossed) for a caller that writes its own hint."""
        was = self._limits_past()
        self.build = self._fit_in(CarBuild.from_json(self.lib.builds[name]))
        cross = self._limit_cross(was, (self.sel,) + tuple(k for k in SLOTS if k != self.sel))
        tail = f" - {cross[0]} is {cross[1]}" if cross else ""
        self.hint = f"loaded build '{self.build.name}'" + (
            "  (default)" if name == self.default_name() else "") + tail
        return tail

    def cycle_build(self, d: int = 1) -> None:
        """B / SHIFT+B: the next / previous of this car's saved builds, in
        place. A car with unsaved wings is not dropped on the first press:
        the hint says so and a second B (the car unchanged) goes on."""
        names = self.car_builds()
        if not names:
            from .prerace import car_label
            self.hint = f"no saved builds for the {car_label(self.car)} yet: S saves this one"
            return
        cur = self.saved_as()
        if cur is None and self.build.has_any(self.lib):
            key = json.dumps(self.build.to_json(), sort_keys=True)
            if getattr(self, "_b_warned", None) != key:
                self._b_warned = key
                self.hint = "unsaved car: B again drops it (S saves it first)"
                return
        self._b_warned = None
        #  where the car in hand stands in the list: the build it IS, else the
        #  one it was loaded as (its name) -- a load that the clamp moved, or
        #  an edit, steps on from there rather than from the top
        at = cur if cur in names else (self.build.name if self.build.name in names else None)
        i = names.index(at) if at is not None else (-1 if d > 0 else 0)
        j = (i + d) % len(names)
        tail = self._load_build(names[j])
        self.hint = f"build {j + 1}/{len(names)}: '{names[j]}'" + (
            "  (default)" if names[j] == self.default_name() else "") + tail

    def _set_default(self, name: str, car: str | None = None) -> bool:
        """Settings.car_build[this car] = `name`, saved -- or `car`'s (task
        46: SAVED CARS' D on another car's build). False with a hint when
        there is no Settings (a bare garage) or the file did not save."""
        st = self.settings
        if st is None or not hasattr(st, "car_build"):
            self.hint = "no drive settings here: set a car's default from the drive's garage"
            return False
        car = car or self.car
        cb = dict(st.car_build) if isinstance(st.car_build, dict) else {}
        if name:
            cb[car] = name
        else:
            cb.pop(car, None)
        st.car_build = cb
        st.save()
        if getattr(st, "save_note", ""):
            self.hint = st.save_note
            return False
        return True

    def make_default(self, name: str | None = None) -> None:
        """F (car page), the menu row, D / R1 (library page): a build becomes
        THIS car's default -- `name`, or the car in hand, saved first when it
        is no saved build yet (as SQUARE saves: in place, or a free name)."""
        from .prerace import car_label
        if self.settings is None:
            self.hint = "no drive settings here: set a car's default from the drive's garage"
            return
        if name is None:
            name = self.saved_as()
            if name is None or not self._own_build(name):
                self._save_build_quick()
                name = self.saved_as()
                if name is None:
                    return                 # the save failed: its hint stands
        if name not in self.lib.builds:
            self.hint = f"'{name}' is not in the library"
            return
        if self._set_default(name):
            c = self.lib.builds[name].get("car", "")
            self.hint = (f"'{name}' is the {car_label(self.car)}'s default build"
                         + (f" (a {car_label(c)} build)" if c and c != self.car else ""))
            self.lib_page.refresh()

    def _rename_build(self, old: str, new: str) -> None:
        """The rename prompt's ENTER: `Library.rename`, and every reference
        to the old name moved with it -- the car in hand, the per-map memory
        (runs/records/last_builds.json, under `records_root`), and any car's
        default in Settings (a default is a NAME, task 41)."""
        try:
            got = self.lib.rename("builds", old, new)
        except KeyError:
            self.hint = f"'{old}' is no longer in the library"
            return
        except (OSError, ValueError) as exc:
            self.hint = _could_not_save(exc)
            return
        if self.build.name == old:
            self.build.name = got
        #  the per-map memory names builds too (review of task 41, finding 9):
        #  it follows, or the next map change drives the old, deleted name
        try:
            from .records import RecordBook
            RecordBook(self.records_root).rename_last_build(old, got)
        except Exception as exc:           # noqa: BLE001 -- the rename itself stands
            print(f"garage: last_builds.json not updated ({type(exc).__name__}: {exc})")
        st = self.settings
        cb = getattr(st, "car_build", None)
        moved = [c for c, n in (cb or {}).items() if n == old] if isinstance(cb, dict) else []
        if moved:
            st.car_build = {c: (got if n == old else n) for c, n in cb.items()}
            st.save()
        self.lib_page.refresh(keep=got)
        self._saved_changed(old)             # task 46: the cards, under the new name
        self.cars_page.set_names(self.cars_page.names, keep=got)
        self.hint = f"build '{old}' renamed '{got}'" + (
            f" (still the default of {len(moved)} car{'s' if len(moved) > 1 else ''})" if moved else "")

    def _new_wing(self) -> None:
        slot = self.build.slot(self.sel)
        slot.wing = ""
        self.build.sync_mirror(self.sel)
        self.open_mission()

    def _delete_library_item(self) -> None:
        """DEL on the library page: the wing or build under the cursor. Task
        45: the first DEL only arms it and says what goes (and when it is a
        car's default, or on the car now); a second DEL on the same item,
        within ARM_S, deletes it (`_delete_saved`, which the card pages of
        task 46 share)."""
        lp = self.lib_page
        it = lp.wings.current() if lp.focus == "wings" else lp.builds.current()
        if self._delete_saved("wings" if lp.focus == "wings" else "builds", it[0] if it else None):
            lp.refresh()

    # -- keyboard / mouse -----------------------------------------------------
    def _handle(self, ev) -> str | None:
        if ev.type == pygame.QUIT:
            return "quit"
        held = False
        if ev.type == pygame.KEYUP:
            self._keys_down.discard(ev.key)
        elif ev.type == pygame.KEYDOWN:
            held = ev.key in self._keys_down
            self._keys_down.add(ev.key)
        if self.prompt.open:
            r = self.prompt.handle(ev)
            if r == "ok":
                self._prompt_done(self.prompt.value)
            return None
        if ev.type == pygame.KEYDOWN and self._armed is not None:
            #  task 45: an armed R / DEL is confirmed by a fresh press of its
            #  own key only -- a held key's auto-repeat is swallowed, any
            #  other key disarms it (and then does what it does)
            if held and ev.key in self._armed["keys"]:
                return None
            if ev.key not in self._armed["keys"]:
                self._disarm()
        if self.menu.open:
            if ev.type == pygame.KEYDOWN:
                cmd = GARAGE_MENU_KEYS.get(ev.key)
                if cmd in ("defaults", "camera"):       # hotkeys close + act
                    self.menu.hide()
                    return self._menu_action(cmd)
                if cmd:
                    return self._menu_action(self.menu.handle(cmd))
                return None
            from .input import _menu_mouse               # the drive's menu mapping:
            cmd = _menu_mouse(ev)                        # point, press, release on
            if cmd:                                      # the row, wheel, right = back
                return self._menu_action(self.menu.handle(cmd))
            return None
        if (self.tutor is not None and ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                                                   pygame.MOUSEWHEEL)
                and self.tutor.hit(getattr(ev, "pos", None) or pygame.mouse.get_pos())):
            return None                                 # the tutorial's box takes its own clicks
        if ev.type == pygame.KEYDOWN:
            #  H is the tutor box's -- except while a number field on the
            #  shell pages is being typed into, where every key is the field's
            if (ev.key == pygame.K_h and self.tutor is not None and self.tutor.active
                    and not (self.page in SHELL_PAGES and self.shell.editing())):
                self.tutor.hidden = not self.tutor.hidden
                return None
        if self.page in SHELL_PAGES:
            #  the mission and design pages are the AeroBO shell's: keys,
            #  clicks, drags and the wheel (design_shell.DesignShell.handle)
            return self.shell.handle(ev)
        if ev.type == pygame.KEYDOWN:
            if self.page == "car":
                return self._handle_car_key(ev)
            return self._handle_page_key(ev)
        if self.page != "car":
            #  THE PAGES TAKE THE MOUSE TOO. They were keyboard-only, which
            #  made a navigator with sixteen steps and a form with twenty-odd
            #  rows a lot of arrow keys.
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self._page_click(ev.pos, bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (4, 5):
                self._page_wheel(1 if ev.button == 4 else -1)
            elif ev.type == pygame.MOUSEWHEEL:
                self._page_wheel(ev.y)
            return None
        if self.page == "car":
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self._drag = True
                elif ev.button == 4:
                    self.cam.zoom(0.90)
                elif ev.button == 5:
                    self.cam.zoom(1.11)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self._drag = False
            elif ev.type == pygame.MOUSEMOTION and self._drag:
                dx, dy = ev.rel
                self.cam.orbit(-dx * 0.008, dy * 0.006)
            elif ev.type == pygame.MOUSEWHEEL:
                self.cam.zoom(0.92 if ev.y > 0 else 1.09)
        return None

    def _handle_car_key(self, ev) -> str | None:
        k = ev.key
        fine = bool(ev.mod & pygame.KMOD_SHIFT)
        if k == pygame.K_ESCAPE:
            self._menu_open()
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "drive"
        elif k == pygame.K_1:
            self._select("left")
        elif k == pygame.K_2:
            self._select("right")
        elif k == pygame.K_3:
            self._select("top")
        elif k == pygame.K_TAB:
            self.sel = SLOTS[(SLOTS.index(self.sel) + 1) % 3]
        elif k == pygame.K_RIGHT:
            self._move(dx=0.01 if fine else STEP_X)
        elif k == pygame.K_LEFT:
            self._move(dx=-(0.01 if fine else STEP_X))
        elif k == pygame.K_UP:
            self._move(dh=0.01 if fine else STEP_H)
        elif k == pygame.K_DOWN:
            self._move(dh=-(0.01 if fine else STEP_H))
        elif k == pygame.K_RIGHTBRACKET:
            self._move(dinc=STEP_INC)
        elif k == pygame.K_LEFTBRACKET:
            self._move(dinc=-STEP_INC)
        elif k == pygame.K_w:
            self._cycle_wing(-1 if fine else 1)
        elif k == pygame.K_m:
            self.build.mirror = not self.build.mirror
            if self.build.mirror:
                self.build.sync_mirror("left")
            self.hint = "left/right mirrored" if self.build.mirror else "left/right independent"
        elif k == pygame.K_t:
            t = self.build.top
            t.mode = "fixed" if t.mode == "active" else "active"
            self.hint = f"top wing deploys: {t.mode}"
        elif k == pygame.K_SPACE:
            self.deploy_cmd = 1.0 - self.deploy_cmd
        elif k == pygame.K_r:
            #  task 45: one R only arms it; a second within ARM_S resets
            if self._confirm("reset", ("R again: remove all three wings (saved builds are kept)"
                                       if self.build.has_any(self.lib) else
                                       "R again: every slot back to its default station"),
                             (pygame.K_r,)):
                self._reset_car()
        elif k == pygame.K_u:
            self._undo_reset()
        elif k == pygame.K_c:
            self.cam.reset()
        elif k == pygame.K_v:
            self.view.show_vectors = not self.view.show_vectors
            self.hint = "force arrows " + ("shown" if self.view.show_vectors else "hidden")
        elif k == pygame.K_d:
            self.open_mission()
        elif k == pygame.K_a:
            self.open_airfoils()
        elif k == pygame.K_l:
            #  task 46: L the SAVED WINGS cards, SHIFT+L the text library
            if fine:
                self.open_library()
            else:
                self.open_saved("wings")
        elif k == pygame.K_g:
            self.open_saved("cars")                   # task 46: SAVED CARS
        elif k == pygame.K_k:
            self.prompt_wing(as_new=fine)             # task 46: save the slot's wing
        #  task 41: the car's builds, one key each. D (the obvious "default")
        #  is the designer's, so the default is F -- the player's Favourite
        elif k == pygame.K_s:
            self.save_build(as_new=fine)
        elif k == pygame.K_b:
            self.cycle_build(-1 if fine else 1)
        elif k == pygame.K_f:
            self.make_default()
        return None

    def _design_shortcut(self, k) -> None:
        """The design page's letter keys (pinned): L screen, O / K optimise /
        keep going, F take the section (PLAN2 §7.3), S put the wing on the
        car, N name it, A the airfoil page. The shell calls this after its
        lock check; a run starts LIVE on AeroBO's worker, and what each one
        did is the model's own log line -- only a refusal is said (a toast on
        the shell pages)."""
        dp = self.design_page
        if k == pygame.K_o:
            self._run_optimiser()
        elif k == pygame.K_k:
            self._run_optimiser(extend=True)
        elif k == pygame.K_l:
            live = self.runs.live
            dp.screen()
            if self.runs.live is live and not dp.nav.current().startswith(("af.", "ep.")):
                self.say(dp.msg)
        elif k == pygame.K_f:
            if not dp.fit_current() and not dp.nav.current().startswith(("af.", "ep.")):
                self.say(dp.msg)
        elif dp.wing is None:
            self.say("state the mission first: the wing is created when the design stages open")
        elif k == pygame.K_s:
            if dp.wing.record is None:
                self.say("no wing yet: run the wing first (3 Wing ▸ Solver, O)", "info")
            else:
                dp.wing.commit()
        elif k == pygame.K_x:
            self.say("the design page's sections are WingLab's: 2 Airfoil ▸ Library screening "
                     "sweeps XFOIL at the surface's own Reynolds number", "info")
        elif k == pygame.K_n:
            self.prompt_rename()
        elif k == pygame.K_a:
            self.open_airfoils()

    def _handle_page_key(self, ev) -> str | None:
        """Keys on the airfoil and library pages (the mission and design
        pages are the shell's)."""
        k = ev.key
        fine = bool(ev.mod & pygame.KMOD_SHIFT)
        if k == pygame.K_ESCAPE:
            self.close_page()
            return None
        if k == pygame.K_UP:
            self._page_nav(-1)
        elif k == pygame.K_DOWN:
            self._page_nav(+1)
        elif k == pygame.K_RIGHT:
            self._page_adjust(+1, fine)
            self._after_rows_changed()
        elif k == pygame.K_LEFT:
            self._page_adjust(-1, fine)
            self._after_rows_changed()
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._page_activate()
            self._after_rows_changed()
        elif k == pygame.K_TAB:
            if self.page == "library":
                self.lib_page.focus = "builds" if self.lib_page.focus == "wings" else "wings"
            elif self.page == "airfoil":
                self.af_page.focus = "params" if self.af_page.focus == "list" else "list"
            elif self.page in ("wings", "cars"):     # task 46: across to the other
                self.open_saved("cars" if self.page == "wings" else "wings")
        elif self.page in ("wings", "cars"):
            self._saved_page_key(k, fine)
        elif self.page == "airfoil":
            if k == pygame.K_x:
                self.af_page.request_xfoil()
            elif k == pygame.K_n:
                self.prompt_naca()
        elif self.page == "library":
            lp = self.lib_page
            if k == pygame.K_s:
                self.prompt_build(as_new=fine)
            elif k == pygame.K_DELETE:              # task 45: never BACKSPACE (the garage key)
                self._delete_library_item()
            elif k == pygame.K_n:
                self._new_wing()
            elif k == pygame.K_d:                     # task 41: the car's default
                if lp.focus == "builds" and lp.builds.current():
                    self.make_default(lp.builds.current()[0])
                else:
                    self.hint = "D makes a BUILD the car's default: TAB to the builds"
            elif k == pygame.K_r:                     # task 41: rename a build
                if lp.focus == "builds":
                    self.prompt_rename_build()
                else:
                    self.hint = "R renames a BUILD (TAB to the builds); a wing: N in its designer"
        return None

    def _saved_page_key(self, k, fine: bool) -> None:
        """The SAVED WINGS / SAVED CARS pages' own keys (task 46; the arrows,
        ENTER, TAB and ESC are every list page's)."""
        if k == pygame.K_l and fine:
            self.open_library()                       # the text library
            return
        if k == pygame.K_DELETE:                      # never BACKSPACE (the garage key)
            if self.page == "wings":
                self._delete_saved("wings", self.wings_page.current())
            else:
                self._delete_saved("builds", self.cars_page.current())
            return
        if self.page == "wings":
            wp = self.wings_page
            if k in (pygame.K_1, pygame.K_2, pygame.K_3):
                self._select(SLOTS[(pygame.K_1, pygame.K_2, pygame.K_3).index(k)])
            elif k == pygame.K_g:
                self.open_saved("cars")
            elif k == pygame.K_k:
                self.prompt_wing(as_new=fine)
            elif k == pygame.K_n:
                self._new_wing()
            elif k == pygame.K_r:
                self.hint = ("a wing is not renamed (builds carry it by name): K on the car "
                             "page saves it under a new one")
            elif k == pygame.K_d:
                self.hint = "D makes a BUILD a car's default: TAB to the saved cars"
            return
        cp = self.cars_page
        if k == pygame.K_s:
            self.prompt_build(as_new=fine)
        elif k == pygame.K_d:
            self._make_default_saved(cp.current())
        elif k == pygame.K_r:
            self.prompt_rename_build(cp.current())
        elif k == pygame.K_l:
            self.open_saved("wings")

    # -- drawing ------------------------------------------------------------------
    def _draw_page(self) -> None:
        if self.page in SHELL_PAGES:
            #  the AeroBO shell: its own chrome, keys and hints (the key-hint
            #  bar below is the dark pages' only)
            self.shell.draw(self.screen)
            return
        u = self.view.ui
        pad_name = self.pad.name if self.pad is not None else None
        if self.page == "car":
            dep = self.deploy * self.deploy * (3.0 - 2.0 * self.deploy)
            self.status_shown = ""            # task 45: never on the car page
            self.view.draw(self.build, self.lib, self.cam, dep, self.sel, pad_name, self.hint,
                           self.hint_alpha(), unlimited=self.unlimited,
                           avoid=self.tutor._rect if self.tutor is not None else None,
                           header=self.build_header())
            return
        self.screen.fill(C_BG)
        help_ = ""
        if self.page == "airfoil":
            help_ = self.af_page.draw(self.screen, self.text, self.plot, u)
            if self._af_for_wing():
                hints = [("CLICK", "a section (twice to use it)"), ("UP/DOWN", "section"),
                         ("TAB", "list / weights"), ("ENTER", "use this section"),
                         ("X", "XFOIL polar"), ("N", "new NACA"), ("ESC", "back")]
                pad_h = [("stick", "section"), ("CROSS", "use"), ("SQUARE", "XFOIL"),
                         ("TRIANGLE", "list/weights"), ("CIRCLE", "back")] if pad_name else None
            else:
                #  task 45: opened from the car page, with no wing being
                #  designed -- browsing only, so the bar promises no ENTER
                #  and says how a section gets used instead
                hints = [("CLICK", "a section"), ("UP/DOWN", "section"),
                         ("TAB", "list / weights"), ("X", "XFOIL polar"), ("N", "new NACA"),
                         ("ESC", "back"), ("ESC then D", "design a wing to use one")]
                pad_h = [("stick", "section"), ("SQUARE", "XFOIL"), ("TRIANGLE", "list/weights"),
                         ("CIRCLE", "back"), ("CIRCLE then L3", "design a wing to use one")] \
                    if pad_name else None
        elif self.page == "wings":
            #  task 46: SAVED WINGS
            self.wings_page.draw(self.screen, self.text, u)
            hints = [("CLICK / ARROWS", "a wing"), ("ENTER", "fit it"), ("1 2 3", "slot"),
                     ("K", "save the slot's wing"), ("TAB", "saved cars"), ("N", "new wing"),
                     ("DEL twice", "delete"), ("SHIFT+L", "as a list"), ("ESC", "back")]
            pad_h = [("stick", "a wing"), ("CROSS", "fit it"), ("L1 / R1", "slot"),
                     ("SQUARE", "save the slot's wing"), ("TRIANGLE", "saved cars"),
                     ("CIRCLE", "back")] if pad_name else None
        elif self.page == "cars":
            #  task 46: SAVED CARS
            self.cars_page.draw(self.screen, self.text, u)
            hints = [("CLICK / ARROWS", "a car"), ("ENTER", "load it"), ("S", "save this car"),
                     ("D", "its car's default"), ("R", "rename"), ("DEL twice", "delete"),
                     ("TAB", "saved wings"), ("SHIFT+L", "as a list"), ("ESC", "back")]
            pad_h = [("stick", "a car"), ("CROSS", "load it"), ("SQUARE", "save this car"),
                     ("R1", "its car's default"), ("TRIANGLE", "saved wings"),
                     ("CIRCLE", "back")] if pad_name else None
        else:
            self.lib_page.draw(self.screen, self.text, u)
            hints = [("UP/DOWN", "item"), ("TAB", "wings / builds"), ("ENTER", "use / load"),
                     ("S", "save car as build"), ("D", "car default"), ("R", "rename"),
                     ("N", "new wing"), ("DEL twice", "delete"), ("ESC", "back")]
            pad_h = [("stick", "item"), ("TRIANGLE", "wings/builds"), ("CROSS", "use / load"),
                     ("SQUARE", "save build"), ("R1", "car default"),
                     ("CIRCLE", "back")] if pad_name else None
        title = {"airfoil": "AIRFOIL LIBRARY", "library": "WING & BUILD LIBRARY",
                 "wings": "SAVED WINGS", "cars": "SAVED CARS"}[self.page]
        r = ui.key_hint_bar(self.screen, self.text, (int(12 * u), int(690 * u), int(1256 * u), int(98 * u)),
                            hints, pad_h, title=f"{title}   {self.build.summary(self.lib)[:100]}")
        if help_:
            self.text.blit(self.screen, help_[:150], r.x + 10, r.bottom - 20, 12, ui.C_KEY)
        if self.page in ("wings", "cars"):
            #  task 46: where the cards' view is, when they do not all fit
            mt = (self.wings_page if self.page == "wings" else self.cars_page).more_text()
            if mt:
                self.text.blit(self.screen, mt, r.right - 10, r.y + 8, 12, C_TEXT_DIM, right=True)
        #  task 46: the card pages' notes -- a refusal with its reason, a fit,
        #  a change of car -- on the bar's last lines (two at most, wrapped
        #  to the bar), gone after HINT_S as everywhere else
        self.saved_hint_drawn = []
        if self.page in ("wings", "cars") and self.hint and self.hint_alpha() > 0.0:
            #  under the bar's rows (`key_hint_bar`: the title at +6, a row
            #  each 17 px from +24)
            y0 = r.y + (60 if pad_h else 43)
            lines = _wrap_lines(self.text, self.hint, r.w - 20, 12, 2)
            for j, ln in enumerate(lines):
                self.text.blit(self.screen, ln, r.x + 10, y0 + j * 14, 12, C_PANEL_ON)
            self.saved_hint_drawn = lines
        #  task 45: the airfoil page's own notes -- ENTER with no wing being
        #  designed, a polar queued -- were written to the hint and drawn
        #  nowhere on it; a line of the bar above the row help, gone after
        #  HINT_S as everywhere else
        if self.page == "airfoil" and self.hint and self.hint_alpha() > 0.0:
            self.text.blit(self.screen, self.hint[:150], r.x + 10, r.bottom - 38, 12, C_PANEL_ON)
        #  the wing-data status: on the DESIGN and AIRFOIL pages, where the
        #  section it is computing for is chosen (task 45)
        self.status_shown = self.status if self.page in ("section", "airfoil") else ""
        if self.status_shown:
            self.text.blit(self.screen, self.status_shown, r.right - 10, r.y + 6, 12, C_TEXT_DIM,
                           right=True)

    # -- the loop -------------------------------------------------------------
    def frame(self, dt: float) -> str | None:
        """One frame: events, pad, deploy animation, XFOIL results, draw."""
        action = None
        self._t_alive += dt
        for ev in pygame.event.get():
            a = self._handle(ev)
            if a:
                action = a
        self._frames += 1
        if self._frames % 30 == 1 and not self.headless:
            try:
                n = pygame.joystick.get_count() if pygame.joystick.get_init() else 0
            except pygame.error:
                n = 0
            if self.pad is None and n > 0:
                self._attach_pad()
            elif self.pad is not None and n == 0:
                self.pad = None
        a = self._poll_pad(dt)
        if a and not action:
            action = a
        if self._pending_action and not action:
            #  task 46: SAVED CARS' change of car, from a key, a click or the pad
            action, self._pending_action = self._pending_action, None
        if action == "drive" and self._t_alive < 0.35:
            action = None                  # a key / button carried in from the drive
        if action == "drive" and self.page != "car":
            self.close_page()
        if self.tutor is not None:
            self.tutor.update(self, action)
        if self._armed is not None and self._t_alive - self._armed["t"] > ARM_S:
            self._disarm()                 # task 45: an unconfirmed R / DEL lapses
        # deploy preview: the same 0.45 s / 0.30 s actuator as vehicle.py
        if self.deploy_cmd > self.deploy:
            self.deploy = min(1.0, self.deploy + dt / 0.45)
        elif self.deploy_cmd < self.deploy:
            self.deploy = max(0.0, self.deploy - dt / 0.30)
        # XFOIL results arriving from the worker
        for name, reb, okp in self.lib.poll():
            if okp:
                for w in self.lib.wings.values():
                    #  carsim-analysed wings only: an AeroBO wing's law is
                    #  AeroBO's, whatever carsim's polar cache learns
                    if w.airfoil == name and not w.legacy and w.engine != "aerobo":
                        ride = w.aero.get("ride_h")
                        self.lib.analyse_wing(w, ride_h=ride)
                        try:
                            self.lib.save_wing(w)
                        except (OSError, ValueError) as exc:
                            self.hint = _could_not_save(exc)
                self.af_page.on_polar(name)
                self.lib_page.refresh()
                self.log(f"XFOIL polar for {name} arrived (Re {reb:.2g}) — the wing re-flies it", "ok")
                if self.page not in SHELL_PAGES:
                    self._bg_hint(f"wing data ready: {name}")
            else:
                self.log(f"XFOIL did not converge for {name}: estimate kept", "warn")
                if self.page not in SHELL_PAGES:
                    self._bg_hint(f"no better wing data for {name}: its estimate stays")
        #  task 45: the notes and the status in a player's words -- the
        #  solver's name and its Reynolds number meant nothing to one -- and
        #  the status only where a section is chosen (`_draw_page`); nothing
        #  when no section is being computed
        busy = self.lib.xfoil_busy
        self.status = (f"computing wing data... ({max(1, self.lib.xfoil_pending)} left)"
                       if busy else "")
        #  the live runs: a unit or two of the current search, inside the
        #  frame's budget (design_jobs.RunManager.pump)
        self.runs.pump(RUN_BUDGET_S)
        self._draw_page()
        if self.tutor is not None:
            self.tutor.draw(self)
        self.menu.draw(self.screen)
        self.prompt.draw(self.screen, self.text)
        return action

    def run(self) -> str:
        pygame.key.set_repeat(260, 45)
        if self.pad is None and not self.headless:
            self._attach_pad()
        clock = pygame.time.Clock()
        try:
            while True:
                dt = min(clock.tick(60) / 1000.0, 0.1)
                action = self.frame(dt)
                pygame.display.flip()
                if action:
                    self.build.clamp(self.lib, self.car)
                    if action == "drive":
                        #  the car drives the laws of the slots as they stand
                        self.rederive_stale()
                    return action
        finally:
            pygame.key.set_repeat()


# =========================================================================== #
#  SELF-CHECK                                                                  #
# =========================================================================== #
class _FakeSettings:
    """What the garage reads of the drive's Settings for the builds (task 41):
    the per-car defaults and a save() that counts."""

    def __init__(self):
        self.car_build, self.saves, self.save_note = {}, 0, ""

    def build_of(self, car=None):
        return self.car_build.get(car or "corsa", "")

    def save(self):
        self.saves += 1


def _check_builds(g: "Garage", lib: Library, key, rep) -> bool:
    """Task 41's build keys on a headless garage `g` on the Corsa, with a car
    in hand that is the library's 'test-car': S in place vs the name prompt,
    another car's build never written over, SHIFT+S, B's order and its
    unsaved-car guard, F / D / R1 (the default) with and without Settings,
    R (rename, the default moving with it), DEL (the default cleared), the
    pad's SQUARE in place and its CROSS on a prompt, and the menu rows."""
    ok = True

    def chk(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        rep(tag, passed, msg)

    def typed(text):
        g.prompt.value = ""
        for ch in text:
            g._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=0, unicode=ch))
        g._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))

    def tap(k):                       # a press and its release (task 45: DEL twice)
        key(k)
        g._handle(pygame.event.Event(pygame.KEYUP, key=k, mod=0))

    SH = pygame.KMOD_SHIFT
    g.page = "car"
    from .records import RecordBook
    g.records_root = os.path.join(os.path.dirname(lib.root), "records")   # never runs/
    chk("a build saved in the garage is tagged with its car",
        lib.builds["test-car"].get("car") == "corsa" and g.build.car == "corsa",
        str(lib.builds["test-car"].get("car")))
    #  S: in place (no prompt, no copy) on the car's own saved build
    n0 = len(lib.builds)
    g.build.left.inc_deg += 1.0
    g.build.sync_mirror("left")
    key(pygame.K_s)
    from .prerace import _same_build
    in_place = (not g.prompt.open and len(lib.builds) == n0
                and _same_build(lib.builds["test-car"], g.build.to_json()))
    key(pygame.K_s)
    chk("S writes this car's saved build over in place; unchanged, it says so",
        in_place and not g.prompt.open and "nothing changed" in g.hint, g.hint)
    #  S on a car with no saved name: the prompt, on its own name
    g.build.name = "fresh one"
    key(pygame.K_s)
    asked = g.prompt.open and g.prompt.value == "fresh one"
    typed("fresh one")
    key(pygame.K_s, SH)
    as_new = g.prompt.open and g.prompt.value == "fresh one-2"
    g._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode=""))
    chk("S on an unsaved name asks it; SHIFT+S opens on a free new name",
        asked and "fresh one" in lib.builds and as_new and not g.prompt.open
        and "fresh one-2" not in lib.builds, g.hint)
    #  another car's build is never written over, by S or by its typed name
    #  (task 46: the Express's, in the retired bus's role)
    bus_js = dict(CarBuild(name="big van", car="express").to_json())
    lib.save_build(bus_js)
    lib.save_build(dict(CarBuild(name="old any").to_json(), car=""))
    g.build.name = "big van"
    key(pygame.K_s)
    prompted = g.prompt.open
    typed("big van")
    chk("another car's build is saved BESIDE, never over (S asks; the typed name gets -2)",
        prompted and lib.builds["big van"] == bus_js and g.build.name == "big van-2"
        and lib.builds["big van-2"]["car"] == "corsa", g.hint)
    #  an any-car build that is ANOTHER car's default is not this car's to
    #  write over (review of task 41, finding 12): S asks and saves beside
    #  it, SQUARE's quick save takes a free name; it keeps its content and
    #  stays any-car, so the rally car still opens with it unchanged
    shared = dict(CarBuild(name="shared any").to_json(), car="")
    lib.save_build(shared)
    st0, g.settings = g.settings, _FakeSettings()
    g.settings.car_build = {"rally": "shared any"}
    g._load_build("shared any")
    g.build.left.inc_deg += 2.0
    g.build.sync_mirror("left")
    key(pygame.K_s)
    asked_sh = g.prompt.open and g.prompt.value == "shared any"
    typed("shared any")
    hint_sh = g.hint
    beside = g.build.name == "shared any-2" and lib.builds["shared any-2"]["car"] == "corsa"
    g._load_build("shared any")
    g.build.left.inc_deg += 1.0
    g.build.sync_mirror("left")
    g._save_build_quick()
    chk("another car's any-car default is saved beside, never over (S asks, SQUARE a free name)",
        asked_sh and beside and "Halcón's default" in hint_sh and g.build.name == "shared any-3"
        and lib.builds["shared any"] == shared, f"{hint_sh}; then {g.hint}")
    g.settings = st0
    for n_ in ("shared any", "shared any-2", "shared any-3"):
        lib.delete("builds", n_)
    #  the library page: this car's builds, the any-car ones, then the van's
    g.lib_page.refresh()
    order = [it[0] for it in g.lib_page.builds.items]
    tags = {it[0]: it[2] for it in g.lib_page.builds.items}
    chk("the library lists this car's builds, then any-car, then other cars' (tagged, dim)",
        order == ["big van-2", "fresh one", "test-car", "old any", "big van"]
        and tags["big van"] == "Courier" and tags["old any"] == "any car"
        and g.lib_page.other == {"big van"}, f"{order} {tags}")
    #  B / SHIFT+B: this car's own and the any-car ones, in that order, in place
    g._load_build("test-car")
    seen = []
    for _ in range(4):
        key(pygame.K_b)
        seen.append(g.build.name)
    key(pygame.K_b, SH)
    back = g.build.name
    chk("B steps this car's saved builds in the library's order (never another car's); "
        "SHIFT+B steps back",
        seen == ["old any", "big van-2", "fresh one", "test-car"] and back == "fresh one",
        f"{seen} then {back}")
    g.build.top.inc_deg += 2.0                        # an unsaved change
    key(pygame.K_b)
    held = g.build.name == "fresh one" and "unsaved" in g.hint
    key(pygame.K_b)
    chk("B does not drop an unsaved car on the first press; the second goes on",
        held and g.build.name == "test-car", g.hint)
    #  F with no Settings (a bare garage): said, nothing written
    key(pygame.K_f)
    bare = "no drive settings" in g.hint
    st = _FakeSettings()
    g.settings = st
    g.build.left.inc_deg -= 1.0                       # unsaved again: F saves it first
    g.build.sync_mirror("left")
    key(pygame.K_f)
    chk("F: no Settings -> a hint; with them, the car in hand (saved first, in place) is "
        "the car's default",
        bare and st.car_build == {"corsa": "test-car"} and st.saves == 1
        and _same_build(lib.builds["test-car"], g.build.to_json()), f"{st.car_build} {g.hint}")
    #  the library: D on another build, R renames it (the default follows), DEL clears it
    g.open_library()
    g.lib_page.focus = "builds"
    g.lib_page.builds.set_items(g.lib_page.builds.items, keep="fresh one")
    key(pygame.K_d)
    d_ok = st.build_of("corsa") == "fresh one" and "(default)" in dict(
        (it[0], it[2]) for it in g.lib_page.builds.items)["fresh one"]
    #  (the per-map memory names it on one map: the rename moves it there too)
    rb_ = RecordBook(g.records_root)
    rb_.set_last_build("linden", "fresh one", lib.builds["fresh one"], car="corsa")
    rb_.set_last_build("arena", "test-car", lib.builds["test-car"], car="corsa")
    key(pygame.K_r)
    rn_open = g.prompt.open and g.prompt.value == "fresh one"
    typed("Fresh Two")
    lb_ = RecordBook(g.records_root).last_builds()
    rn_ok = ("Fresh Two" in lib.builds and "fresh one" not in lib.builds
             and st.build_of("corsa") == "Fresh Two"
             and g.lib_page.builds.current()[0] == "Fresh Two"
             and (lb_["linden|corsa"]["name"], lb_["linden|corsa"]["build"]["name"])
             == ("Fresh Two", "Fresh Two") and lb_["arena|corsa"]["name"] == "test-car")
    tap(pygame.K_DELETE)
    del_1 = ("Fresh Two" in lib.builds and st.build_of("corsa") == "Fresh Two"
             and g.hint == "DEL again: delete build Fresh Two (the Civetta's default)")
    tap(pygame.K_DELETE)
    chk("library D makes the build under the cursor the default; R renames it (the default "
        "and the per-map memory follow); DEL asks (naming the default), DEL again deletes it "
        "and clears the default",
        d_ok and rn_open and rn_ok and del_1 and "Fresh Two" not in lib.builds
        and st.build_of("corsa") == "" and "none now" in g.hint, g.hint)
    #  the pad: SQUARE saves in place (no '-2' pile), R1 the default, CROSS a prompt
    fp = type("_P", (), dict(name="DualSense Wireless Controller", layout="ps",
                             held=set(), pressed=lambda self, n: n in self.held,
                             stick=lambda self, w="left": (0.0, 0.0)))()
    g.pad, g._pad_seeded = fp, False
    g.frame(1.0 / 60.0)

    def press(btn):
        fp.held.add(btn)
        g.frame(1.0 / 60.0)
        fp.held.discard(btn)
        g.frame(1.0 / 60.0)
    g._load_build("test-car")
    g.page, g.lib_page.focus = "library", "builds"
    n1 = len(lib.builds)
    g.build.left.inc_deg += 1.0
    g.build.sync_mirror("left")
    press("square")
    press("square")
    sq_ok = len(lib.builds) == n1 and _same_build(lib.builds["test-car"], g.build.to_json())
    g.lib_page.builds.set_items(g.lib_page.builds.items, keep="old any")
    press("r1")
    r1_ok = st.build_of("corsa") == "old any"
    g._menu_action("build_save_as")
    cross_ok = g.prompt.open and g.prompt.value == "test-car-2"
    press("cross")
    chk("pad: SQUARE saves in place (no copies), R1 sets the default, CROSS takes a "
        "prompt's name",
        sq_ok and r1_ok and cross_ok and not g.prompt.open and "test-car-2" in lib.builds,
        g.hint)
    g.pad = None
    #  the pause menu's rows: what the pad has for S, SHIFT+S, B and F
    g.page = "car"
    g._menu_open()
    acts = [a for _, a in g.menu.items]
    rows = dict((a, lbl) for lbl, a in g.menu.items)
    g.menu.hide()
    #  task 46: 'Load a build ...' is the Saved cars row now (its old action
    #  still opens that page), beside Saved wings and the slot's Save wing
    g._menu_action("saved_cars")
    on_cars = g.page == "cars"
    g.page = "car"
    g._menu_action("build_load")
    chk("the pause menu has Save / Save as / Saved cars (the loads) / Make default (showing "
        "the default), Saved wings and Save the side wing",
        {"build_save", "build_save_as", "saved_cars", "saved_wings", "wing_save",
         "build_default"} <= set(acts) and "build_load" not in acts
        and "(now: old any)" in rows["build_default"] and on_cars and g.page == "cars"
        and rows["saved_cars"] == "Saved cars" and rows["saved_wings"] == "Saved wings"
        and rows["wing_save"] == f"Save the {'top' if g.sel == 'top' else 'side'} wing as ...",
        rows.get("build_default", ""))
    g.page = "car"
    g.settings = None
    return ok


def _check_confirms(g: "Garage", lib: Library, key, rep) -> bool:
    """Task 45's two presses on a headless garage `g` whose car in hand has
    wings: one R changes nothing (it says what a second does), R R clears
    the car, U puts it back; a held R's auto-repeat, another key between or
    a lapsed ARM_S never confirms; the pause menu's row (ENTER, and its R
    hotkey) the same, its label saying so; on the library page BACKSPACE
    deletes nothing and a build goes on DEL DEL, a wing's first DEL only
    asks."""
    ok = True

    def chk(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        rep(tag, passed, msg)

    def up(k):
        g._handle(pygame.event.Event(pygame.KEYUP, key=k, mod=0))

    def tap(k):
        r_ = key(k)
        up(k)
        return r_

    def js():
        return json.dumps(g.build.to_json(), sort_keys=True)

    g.page, g._armed = "car", None
    g._load_build("test-car")
    j0 = js()
    tap(pygame.K_r)
    one = js() == j0 and g.hint.startswith("R again: remove all three wings")
    tap(pygame.K_r)
    cleared = not g.build.has_any(lib) and "U puts them back" in g.hint
    tap(pygame.K_u)
    back = js() == j0 and g.hint == "wings back" and g._undo_build is None
    tap(pygame.K_u)
    chk("car page: one R changes nothing (the hint asks), R R clears every wing, U puts them "
        "back (once)",
        g.build.has_any(lib) and one and cleared and back and "nothing to undo" in g.hint,
        g.hint)
    key(pygame.K_r)
    key(pygame.K_r)                   # auto-repeat: down again with no release
    rep_ok = js() == j0
    up(pygame.K_r)
    tap(pygame.K_c)                   # another key between: disarmed
    c_hint = g.hint
    tap(pygame.K_r)
    g._t_alive += ARM_S + 0.1         # ... and lapsed
    g.frame(1.0 / 60.0)
    lapsed = g._armed is None and g.hint == ""
    tap(pygame.K_r)
    chk("a held R's auto-repeat, another key between, or ARM_S lapsing never confirms "
        "(the 'again' hint goes)",
        rep_ok and c_hint == "" and lapsed and js() == j0, g.hint)
    tap(pygame.K_c)
    #  the pause menu's row, by ENTER and by its R hotkey
    g._menu_open(at="defaults")
    tap(pygame.K_RETURN)
    rows = dict((a_, lbl) for lbl, a_ in g.menu.items)
    m1 = (g.menu.open and g.menu.action() == "defaults" and js() == j0
          and rows["defaults"].startswith("Reset car: ENTER again"))
    tap(pygame.K_RETURN)
    m2 = not g.menu.open and not g.build.has_any(lib)
    tap(pygame.K_u)
    g._menu_open()
    tap(pygame.K_r)
    h1 = g.menu.open and g.menu.action() == "defaults" and js() == j0
    tap(pygame.K_r)
    h2 = not g.menu.open and not g.build.has_any(lib)
    tap(pygame.K_u)
    g._menu_open()
    tap(pygame.K_r)
    tap(pygame.K_ESCAPE)              # the menu left: disarmed
    tap(pygame.K_r)                   # so R on the car page only asks
    esc = js() == j0 and g.hint.startswith("R again")
    tap(pygame.K_c)
    chk("the menu's 'Reset car' row: ENTER asks (the row says 'ENTER again', cursor kept), "
        "ENTER again resets; its R hotkey the same; ESC disarms it",
        m1 and m2 and h1 and h2 and esc and js() == j0,
        f"{m1} {m2} {h1} {h2} {esc}: {rows.get('defaults')}")
    #  the library: BACKSPACE (the drive's garage key) never deletes
    lib.save_build(dict(CarBuild(name="scrap", car="corsa").to_json()))
    g.open_library()
    g.lib_page.refresh()
    g.lib_page.focus = "builds"
    g.lib_page.builds.set_items(g.lib_page.builds.items, keep="scrap")
    tap(pygame.K_BACKSPACE)
    bs = "scrap" in lib.builds
    tap(pygame.K_DELETE)
    d1 = "scrap" in lib.builds and g.hint == "DEL again: delete build scrap"
    tap(pygame.K_DELETE)
    user = next((n_ for n_ in g.build.wings(lib).values()
                 if n_ is not None and not n_.builtin), None)
    g.lib_page.focus = "wings"
    if user is not None:
        g.lib_page.wings.set_items(g.lib_page.wings.items, keep=user.name)
    tap(pygame.K_DELETE)
    w1 = (user is not None and user.name in lib.wings
          and g.hint == f"DEL again: delete wing {user.name} (on the car now: it comes off)")
    tap(pygame.K_DOWN)
    chk("library: BACKSPACE deletes nothing, DEL asks, DEL DEL deletes a build; a wing's "
        "first DEL asks (it is on the car)",
        bs and d1 and "scrap" not in lib.builds and w1 and user.name in lib.wings, g.hint)
    g.close_page()
    g.page, g._armed = "car", None
    return ok


def _check_hints(g: "Garage", rep) -> bool:
    """Task 45's hints on a headless garage `g`: on the car page a hint,
    short or long, is drawn on its own lines over the bar (never on the
    bar's key lines), whole in two lines, left of the wing tutorial's box
    when that is up; it is gone HINT_S after it was set; and the wing-data
    notes and status are in plain words, the status off the car page."""
    ok = True

    def chk(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        rep(tag, passed, msg)

    u = g.view.ui
    bar = int(690 * u)
    g.page, g._armed = "car", None
    g.lib.poll = lambda: []             # no solver result may land in between
    try:
        short = "'Fast Wings' is the Civetta's default build"
        long_ = ("left: the panel's lower tip is at the ground clearance (h 0.58 m for 0.88 m)"
                 " - Settings > Wing limits: Unlimited goes lower")
        drawn = {}
        for h in (short, long_):
            g.hint = h
            g.frame(1.0 / 60.0)
            drawn[h] = list(g.view.hint_drawn)
        tut = pygame.Rect(*(int(v * u) for v in (748, 468, 520, 212)))
        beside = g.view._hint_lines(long_, avoid=tut)

        def whole(lines, h):
            return " ".join(s_ for s_, _, _ in lines) == " ".join(h.split())

        above = all(0 < len(ls) <= 2 and all(y + 18 * u <= bar for _, _, y in ls)
                    for ls in (*drawn.values(), beside))
        chk("a hint is drawn on its own lines over the bar, short or long, whole in two "
            "lines, and left of the wing tutorial's box",
            above and len(drawn[short]) == 1 and whole(drawn[short], short)
            and whole(drawn[long_], long_) and whole(beside, long_)
            and all(x + g.view.f_lbl.size(s_)[0] <= tut.x for s_, x, _ in beside),
            f"{[(s_[:24], int(x), int(y)) for s_, x, y in drawn[long_]]}; "
            f"beside the box {[(int(x), int(y)) for _, x, y in beside]}")
        g.hint = short
        seen = []
        for _ in range(18):                 # 4.5 s in 0.25 s frames
            g.frame(0.25)
            seen.append((round(g._t_alive - g._hint_t, 2), round(g.hint_alpha(), 2),
                         bool(g.view.hint_drawn)))
        fading = [a for t_, a, _ in seen if HINT_S - HINT_FADE_S < t_ < HINT_S]
        chk("a hint fades over its last HINT_FADE_S and is not drawn 4.5 s after it was set",
            seen[0][2] and not seen[-1][2] and seen[-1][1] == 0.0 and g.hint == short
            and fading and all(0.0 < a < 1.0 for a in fading),
            f"alpha {[a for _, a, _ in seen]}")
        notes = []
        for okp in (True, False):
            g.lib.poll = lambda okp=okp: [("zz-none", 7.0e5, okp)]
            g.frame(1.0 / 60.0)
            g.lib.poll = lambda: []
            notes.append(g.hint)
        g.lib.xfoil_busy = "zz-none Re 7e+05"
        shown = {}
        for pg in ("car", "airfoil"):
            g.page = pg
            g.frame(1.0 / 60.0)
            shown[pg] = g.status_shown
        g.lib.xfoil_busy = ""
        words = notes + [g.status]
        chk("the wing-data notes and status in plain words; the status never on the car page",
            notes[0] == "wing data ready: zz-none"
            and not any("XFOIL" in w_ or "Re " in w_ for w_ in words)
            and shown["car"] == "" and shown["airfoil"] == "computing wing data... (1 left)",
            f"{words}; car {shown['car']!r}")
    finally:
        del g.lib.poll
        g.page, g._armed = "car", None
    return ok


def _check_build_header(g: "Garage", lib: Library, key, rep) -> bool:
    """Task 45's BUILD line and name prompt on a headless garage `g`: the
    car page says which build is in hand, whether it is saved and whether
    it is the car's default, left of the slot panel and SPAN LIMITS; a name
    prompt opens on its name SELECTED -- a key replaces it, a first
    BACKSPACE clears it, ENTER untouched takes it -- and its key line never
    runs past its box."""
    ok = True

    def chk(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        rep(tag, passed, msg)

    def kd(k, ch="", mod=0):
        return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=ch)

    u = g.view.ui
    st0, b0, held0 = g.settings, CarBuild.from_json(g.build.to_json()), g._held
    had = "Fast Wings" in lib.builds
    g.page, g._armed = "car", None
    g.lib.poll = lambda: []             # no solver result may land in between
    try:
        #  the line: saved and the default, then an edit, then no name
        g.settings = _FakeSettings()
        g.settings.car_build = {g.car: "test-car"}
        g._load_build("test-car")
        g.frame(1.0 / 60.0)
        saved, box = g.view.header_drawn
        g.build.left.inc_deg += 1.0
        g.build.sync_mirror("left")
        g.frame(1.0 / 60.0)
        edited = g.view.header_drawn[0]
        g.build.name = ""
        unnamed = "".join(s_ for s_, _ in g.build_header())
        chk("the car page's BUILD line: the build, 'saved' and the car's default; an edit "
            "'unsaved changes'; no name '(unnamed)'; left of the slot panel and SPAN LIMITS",
            saved.startswith("BUILD  test-car") and "saved" in saved and "default" in saved
            and "unsaved" not in saved and "unsaved changes (S saves)" in edited
            and "default" not in edited and "(unnamed)" in unnamed
            and box is not None and box.right < int(884 * u) and box.bottom < int(490 * u),
            f"{saved!r}; {edited!r}; {unnamed!r}; box {tuple(box) if box else None}")
        #  the prompt: its preset is selected
        p = ui.TextPrompt()
        p.show("save the current car as a build", "my corsa")
        p.handle(kd(pygame.K_LSHIFT))                  # a modifier alone keeps it selected
        for ch in "Fast":
            p.handle(kd(pygame.K_a, ch))
        typed = p.value
        p.show("save the current car as a build", "my corsa")
        p.handle(kd(pygame.K_BACKSPACE))
        bs = p.value
        for ch in "xy":
            p.handle(kd(pygame.K_a, ch))
        p.handle(kd(pygame.K_BACKSPACE))
        bs2 = p.value                                  # after that, one character
        p.show("save the current car as a build", "my corsa")
        enter = (p.handle(kd(pygame.K_RETURN, "\r")), p.value)
        p.show("save the current car as a build", "my corsa")
        p.handle(kd(pygame.K_RIGHT))
        p.handle(kd(pygame.K_a, "2"))
        kept = p.value
        p.handle(kd(pygame.K_BACKSPACE, mod=pygame.KMOD_CTRL))
        chk("a name prompt opens on its name selected: 'Fast' typed gives 'Fast', a first "
            "BACKSPACE '', ENTER untouched the name; then BACKSPACE takes one character, "
            "RIGHT keeps the name to type on, CTRL+BACKSPACE clears",
            typed == "Fast" and bs == "" and bs2 == "x" and enter == ("ok", "my corsa")
            and kept == "my corsa2" and p.value == "",
            f"{typed!r} {bs!r} {bs2!r} {enter} {kept!r} {p.value!r}")
        #  ... and through the garage: S on a car whose name the library
        #  lacks asks on that name; 'Fast Wings' typed is the name it saves
        g.build.name = "hdr one"
        key(pygame.K_s)
        asked = g.prompt.open and g.prompt.selected and g.prompt.value == "hdr one"
        for ch in "Fast Wings":
            g._handle(kd(pygame.K_SPACE if ch == " " else pygame.K_a, ch))
        in_box = g.prompt.value
        g._handle(kd(pygame.K_RETURN, "\r"))
        g.frame(1.0 / 60.0)
        now = g.view.header_drawn[0]
        chk("S asks on the car's name, selected; 'Fast Wings' typed is saved as 'Fast Wings' "
            "and the line then says it is saved",
            asked and in_box == "Fast Wings" and "Fast Wings" in lib.builds
            and "hdr one" not in lib.builds and now.startswith("BUILD  Fast Wings   saved"),
            f"asked {asked}; box {in_box!r}; line {now!r}")
        #  the key line inside the box, however long (the swarm's names its bot)
        fits = []
        for hint in (Garage.PROMPT_HINT,
                     "ENTER save (empty = swarm-arena-corsa-2026-09-26-1432)   "
                     "ESC discard - no checkpoint",
                     "   ".join(["ENTER / CROSS ok   ESC / CIRCLE cancel"] * 5)):
            p.show("save the current car as a build", "my corsa", hint)
            r_, lines = p.layout(g.screen, g.text)
            fits.append((r_.w, len(lines),
                         all(g.text.width(s_, 12) <= r_.w - 2 * int(16 * u) for s_ in lines)
                         and r_.w <= int(720 * u) and 0 <= r_.x and r_.right <= g.screen.get_width()))
        chk("a name prompt's key line fits its box (widened up to 720 px, then wrapped)",
            all(f_ for _, _, f_ in fits) and fits[-1][1] > 1, str(fits))
    finally:
        del g.lib.poll
        g.prompt.open_ = False
        if "Fast Wings" in lib.builds and not had:
            lib.delete("builds", "Fast Wings")     # the library as it came in
        g.settings, g.build, g._held = st0, b0, held0
        g.lib_page.refresh()
        g.page, g._armed = "car", None
    return ok


def _check_menu_fits(lib: Library, pad, rep) -> bool:
    """Task 45: the pause menu laid out at 1280x800 -- the keyboard's help
    alone, the keyboard's and `pad`'s together, and the same with the
    widest rows the menu can have: 24-character build and default names on
    the Express, and the wing tutorial running (its three rows make the
    list scroll, its 'v 16 more' widening the rows). Then at 1280x720,
    1440x900 and 1600x900, where the font steps are not the window's
    (13 px text at 0.9 of 1280x800, 16 px at 1.125) and the help wrapped
    and ran past the footer: 'my express' on the Express, the wing tutorial
    on, keyboard and keyboard + pad. Each fits with margin: the panel 24 px
    or more inside the window, every text inside the panel, no help line
    wrapped, the last help line 16 px or more above the footer. No help text
    is longer than HELP_TEXT_MAX, and a long name is cut to MENU_NAME_MAX."""
    from .wing_tutorial import WingTutor
    ok = True
    long_name = "all four wings, wet map"[:24].ljust(24, "!")

    def lay(size, with_pad, name, tutor, done=False):
        """The menu drawn once at `size`: (fits, what was measured). `done`:
        the wing tutorial finished -- its row, the widest, stands in the list
        (and Settings, so the Change car row, are there too)"""
        st = type("_St", (), dict(build_of=lambda self, car: name))() if name else None
        if done and st is None:
            st = type("_St", (), dict(build_of=lambda self, car: ""))()
        g = Garage(size, CarBuild(), pad=pad if with_pad else None, headless=True,
                   lib=lib, car="express" if name else "corsa", settings=st)
        if done:
            from .wing_tutorial import SECTION
            g.progress = type("_Pr", (), dict(section=lambda self, s_: (
                {"done": True, "offered": True} if s_ == SECTION else {})))()
        if name:
            g.build.name = name
        if tutor:
            g.tutor = WingTutor(None)
            g.tutor.i = 3                 # step 4, the longest title
        g._menu_open()
        g.menu.draw(g.screen)
        m, win = g.menu, g.screen.get_rect()
        p, texts = m._last_panel, m._last_text_rects
        foot = [r for t, r in texts if t == m.footer]
        low = max(r.bottom for t, r in texts if t != m.footer)
        drawn = {t for t, _ in texts}
        wraps = sum(1 for _, rows in m.sections for _, w in rows if w and w not in drawn)
        rows = dict((a, lbl) for lbl, a in m.items)
        shown = (name if len(name or "") <= MENU_NAME_MAX
                 else name[:MENU_NAME_MAX - 3] + "...")
        fits = (p is not None and bool(foot) and p.top - win.top >= 24
                and win.bottom - p.bottom >= 24
                and all(p.left <= r.left and r.right <= p.right - 8 and p.top <= r.top
                        and r.bottom <= p.bottom for _, r in texts)
                and wraps == 0 and foot[0].top - low >= 16
                and (m.sections[-1][0] == "PS5 DUALSENSE") == with_pad
                and (not name or ("'" + shown + "'" in rows["build_save"]
                                  and rows["build_default"].startswith("Set as Courier default")
                                  and "(now: " + shown + ")" in rows["build_default"]))
                and (not tutor or "wt_skip" in rows)
                and (not done or ("wt_start" in rows and "cars" in rows)))
        return fits, (f"panel y {p.top if p else None}..{p.bottom if p else None} of "
                      f"{win.height}, text to y {low}, footer at y "
                      f"{foot[0].top if foot else None}, {len(m.items)} rows, "
                      f"{wraps} help line(s) wrapped")

    cases = (("keyboard", False, "", False), ("keyboard + pad", True, "", False),
             ("keyboard + pad, 24-character names", True, long_name, False),
             ("keyboard + pad, names, wing tutorial on", True, long_name, True),
             ("keyboard, names, wing tutorial on", False, long_name, True))
    for label, with_pad, name, tutor in cases:
        fits, got = lay((1280, 800), with_pad, name, tutor)
        ok = ok and fits
        rep(f"the garage menu fits at 1280x800 with margin: {label}", fits, got)
    #  task 45 (round 3): the sizes it did not fit at, the wing tutorial on
    #  and a 10-character name, whole, in the build rows
    for size in ((1280, 720), (1440, 900), (1600, 900)):
        got = [lay(size, with_pad, "my courier", True) for with_pad in (False, True)]
        fits = all(f_ for f_, _ in got)
        ok = ok and fits
        rep(f"the garage menu fits at {size[0]}x{size[1]} with margin, the wing tutorial on "
            "and 'my courier' on the Courier: keyboard; keyboard + pad", fits,
            "; ".join(d_ for _, d_ in got))
    #  Change car (review 1): the drive's garage (Settings: its row) of a
    #  player who has FINISHED the wing tutorial -- that row the widest -- at
    #  every size, keyboard and pad
    for size in ((1280, 800), (1280, 720), (1440, 900), (1600, 900)):
        got = [lay(size, with_pad, "my courier", False, done=True) for with_pad in (False, True)]
        fits = all(f_ for f_, _ in got)
        ok = ok and fits
        rep(f"the garage menu fits at {size[0]}x{size[1]} with margin, Change car's row and a "
            "finished wing tutorial's row: keyboard; keyboard + pad", fits,
            "; ".join(d_ for _, d_ in got))
    longest = max(len(w) for _, w in GARAGE_HELP_KB + GARAGE_HELP_PAD)
    short = (longest <= HELP_TEXT_MAX and _menu_name("my express") == "my express"
             and len(_menu_name(long_name)) == MENU_NAME_MAX)
    ok = ok and short
    rep("the menu's help texts are HELP_TEXT_MAX characters or fewer; a build's name in its "
        "rows is cut to MENU_NAME_MAX ('my express' whole)", short,
        f"longest help text {longest} (max {HELP_TEXT_MAX}); {len(GARAGE_HELP_KB)} keyboard + "
        f"{len(GARAGE_HELP_PAD)} pad rows; '{_menu_name(long_name)}'")
    return ok


class _Settings46:
    """What task 46's rows read of the drive's Settings: the per-car
    defaults, the wing limits, the paint names and a save() that counts."""

    def __init__(self, limits: str = "real"):
        self.car_build, self.saves, self.save_note = {}, 0, ""
        self.wing_limits = limits

    def build_of(self, car=None):
        return self.car_build.get(car or "corsa", "")

    def save(self):
        self.saves += 1


def _check_saved(g0: "Garage", lib0: Library, rep, screenshot_dir: str = "") -> bool:
    """Task 46, on a library of its own beside `lib0`'s (the owner's shapes:
    a 1.88 m WingLab-style side wing made for the Express, a 0.80 m one of
    the player's, 'my corsa' and 'my express (autosave)'):

      * `WingSpec.made_for`: an old file decodes "", the tag survives a
        round trip, a copy keeps it;
      * K / SHIFT+K / the menu row: refused on an empty slot and on the
        published panels; in place on its own name (the car recorded once);
        a new name a copy (design and aero kept, not built in, made on this
        car) the slot then carries, mirrored; a taken name saved beside;
        a built-in's copy offered 'my ...'; the pad's CROSS takes the name;
      * `wing_fit`: the role, this car's limit at the slot (the rule W
        holds, wing for wing), x3 in Unlimited; the reasons in words;
      * SAVED WINGS: L opens it, the player's own first, each card's fit as
        `wing_fit` says, dimmed with its reason; ENTER fits one that fits
        (and goes to the car page), refuses one that does not with the
        reason and what would fit it; 1 / 2 / 3 and a slot chip change the
        slot; the mouse (a click selects, a second fits); DEL twice; TAB;
        the diagram's scale, plates, pylons and dimming; the page's budget;
      * SAVED CARS: G opens it, pictures drawn a few a frame, cached (the
        same surface next frame), drawn afresh after a save / rename /
        delete; ENTER loads this car's build, asks the drive for another
        car's (car_wanted + build_wanted, 'car' out of the frame), refuses
        in a bare garage or on a challenge's car; D sets ITS car's default;
        R renames; the drive's `_garage_car` opens the car on that build."""
    import copy as _copy
    import shutil
    ok = True

    def chk(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        rep(tag, passed, msg)

    root = os.path.join(os.path.dirname(lib0.root.rstrip(os.sep)), "library46")
    shutil.rmtree(root, ignore_errors=True)
    lib = Library(root, use_xfoil=False)
    base = lib.wings["flank-e423"]
    big = base.copy(name="big side", builtin=False, span=1.88, chord=0.70, taper=0.62,
                    notes="a WingLab-sized side wing", design={"engine": "carsim-test"})
    big.aero = dict(base.aero)
    lib.save_wing(big)
    mine = base.copy(name="my side", builtin=False, made_for="corsa")
    lib.save_wing(mine)
    carried = base.copy(name="carried side", builtin=False, span=0.64, chord=0.25, mount="endplate",
                        plate_h=0.45, plate_cant_deg=70.0)
    lib.save_wing(carried)
    js_c = CarBuild.for_car("corsa")
    js_c.name, js_c.car = "my corsa", "corsa"
    js_c.left.wing = "flank-e423"
    js_c.sync_mirror("left")
    js_c.top.wing = "rear-s1223"
    lib.save_build(js_c.to_json())
    js_e = CarBuild.for_car("express")
    js_e.name, js_e.car = "my express (autosave)", "express"
    js_e.left.wing = "big side"
    js_e.sync_mirror("left")
    lib.save_build(js_e.to_json())

    # --- the WingSpec's new label ---------------------------------------------
    from .aero.wing import WingSpec as _WS
    old = {k: v for k, v in big.to_json().items() if k != "made_for"}
    tagged = big.copy(made_for="express")
    chk("WingSpec.made_for: an old wing file decodes as '' (not recorded), the tag survives a "
        "round trip and a copy, a non-string tag reads ''",
        _WS.from_json(old).made_for == "" and _WS.from_json(tagged.to_json()).made_for == "express"
        and tagged.copy(name="x").made_for == "express"
        and _WS.from_json(dict(old, made_for=7)).made_for == ""
        and set(_WS.from_json(old).to_json()) == set(old) | {"made_for"}
        and _WS.from_json(old).to_json() == dict(_WS.from_json(tagged.to_json()).to_json(),
                                                 made_for=""),
        f"{_WS.from_json(tagged.to_json()).made_for!r}")
    chk("a wing's origin: recorded ('made on the Civetta'), else the one car whose builds carry it "
        "('used on the Courier'), a built-in's 'on every car', else 'not recorded'",
        origin_text(lib, "my side") == "made on the Civetta"
        and origin_text(lib, "big side") == "used on the Courier"
        and origin_text(lib, "plate") == "built in: on every car"
        and origin_text(lib, "carried side") == "car not recorded",
        f"{origin_text(lib, 'big side')!r}")

    # --- fits, and the reasons -------------------------------------------------
    real = _Settings46("real")
    g = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, car="corsa",
               settings=real)
    g.records_root = os.path.join(root, "records")      # a rename's per-map memory: here
    b90 = g.build                                  # the Corsa's side slot at h 0.90
    f_big = wing_fit(big, "left", b90, "corsa")
    f_top = wing_fit(lib.wings["rear-s1223"], "left", b90, "corsa")
    f_side = wing_fit(mine, "top", b90, "corsa")
    f_unl = wing_fit(big, "left", b90, "corsa", unlimited=True)
    huge = big.copy(name="huge", span=5.0)
    f_huge = wing_fit(huge, "left", b90, "corsa", unlimited=True)
    exp = CarBuild.for_car("express")
    f_exp = wing_fit(big.copy(span=1.70), "left", exp, "express")
    chk("wing_fit: a 1.88 m side wing is too wide for the Corsa's side slot at h 0.90 (in words), "
        "fits in Unlimited (past: its runs UNLIMITED), a 5 m one not even then; a top wing "
        "wants the top slot and a side wing a side slot; a 1.70 m one fits the Express",
        f_big[:2] == (False, "too wide for the Aurel Civetta: 1.88 m, the limit here is 1.50 m")
        and f_top[:2] == (False, "a top wing: select the top slot")
        and f_side[:2] == (False, "a side wing: select a side slot")
        and f_unl[0] and f_unl[3] and not f_huge[0] and "even in Unlimited" in f_huge[1]
        and "4.50 m" in f_huge[1] and f_exp[0] and not f_exp[3]
        and abs(fit_height(big, "corsa") - 1.09) < 1e-9 and fit_height(huge, "corsa") is None,
        f"{f_big[1]!r}; {f_huge[1]!r}")
    same = []
    for h in (0.55, 0.75, 0.90, 1.05, 1.20):
        g.build.left.h = h
        for n, w in lib.wings.items():
            if w.role == "flank":
                same.append(wing_fit(w, "left", g.build, "corsa")[0]
                            == (g._past_limit("left", w) is None))
    chk("wing_fit holds W's rule in Real mode: wing for wing and height for height, it fits "
        "exactly when W's span limit lets it in", all(same) and len(same) >= 25,
        f"{sum(same)}/{len(same)} agree")
    g.build.left.h = 0.90
    g.build.sync_mirror("left")

    # --- K: save the slot's wing --------------------------------------------
    def keyd(k, mod=0, ch=""):
        return g._handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=ch))

    def typed(text):
        g.prompt.value, g.prompt.selected = "", False
        for ch_ in text:
            keyd(ord(ch_) if ch_.isalnum() else pygame.K_SPACE, 0, ch_)
        keyd(pygame.K_RETURN, 0, "\r")

    g.page, g.sel = "car", "left"
    keyd(pygame.K_k)
    empty = (not g.prompt.open, g.hint)
    g.build.left.wing = "plate"
    g.build.sync_mirror("left")
    keyd(pygame.K_k)
    legacy = (not g.prompt.open, g.hint)
    g.build.left.wing = "flank-e423"
    g.build.sync_mirror("left")
    keyd(pygame.K_k)
    offered = g.prompt.value
    keyd(pygame.K_RETURN, 0, "\r")
    bi = lib.wings.get("my low-drag side wing")
    chk("K: refused on an empty slot and on the published panels (on every car already); on a "
        "built-in it offers a free 'my ...' name, and ENTER saves a copy there",
        empty[0] and "no side wing in the left slot" in empty[1] and legacy[0]
        and "published panel" in legacy[1] and offered == "my low-drag side wing"
        and bi is not None and not bi.builtin and bi.made_for == "corsa"
        and g.build.left.wing == g.build.right.wing == "my low-drag side wing",
        f"{empty[1]!r}; {legacy[1]!r}; offered {offered!r}")
    g.build.left.wing = "big side"
    g.build.sync_mirror("left")
    g.build.left.h = 1.15                      # (the Express build's height: it fits there)
    g.build.sync_mirror("left")
    keyd(pygame.K_k)
    own_name = g.prompt.value
    keyd(pygame.K_RETURN, 0, "\r")
    in_place = lib.wings["big side"].made_for == "corsa" and "big side" in g.hint
    keyd(pygame.K_k)
    keyd(pygame.K_RETURN, 0, "\r")
    again = lib.wings["big side"].made_for == "corsa" and "nothing changed" in g.hint
    lib.wings["big side"].made_for = "express"          # as WingLab would have stamped it
    lib.save_wing(lib.wings["big side"])
    g.sel = "right"
    keyd(pygame.K_k, pygame.KMOD_SHIFT)
    free = g.prompt.value
    typed("express wing")
    cp = lib.wings.get("express wing")
    back = Library(root, use_xfoil=False).wings.get("express wing")
    chk("K on a wing of the player's: its own name, ENTER saves it in place (the car recorded once, "
        "then 'nothing changed'); SHIFT+K a free name; a typed one a COPY -- not built in, "
        "design and aero kept, made on this car -- the slot carries, both sides mirrored; "
        "it is on disk",
        own_name == "big side" and in_place and again and free == "big side-2"
        and cp is not None and not cp.builtin and cp.made_for == "corsa"
        and cp.design == big.design and cp.aero == big.aero and abs(cp.span - 1.88) < 1e-12
        and g.build.left.wing == g.build.right.wing == "express wing"
        and back is not None and back.made_for == "corsa" and back.design == big.design,
        g.hint)
    keyd(pygame.K_k, pygame.KMOD_SHIFT)
    typed("my side")                                # another wing holds it
    taken = (lib.wings["my side"].span == mine.span and "my side-2" in lib.wings
             and "'my side' is taken" in g.hint and g.build.left.wing == "my side-2")
    g._menu_open()
    rows = dict((a, lbl) for lbl, a in g.menu.items)
    g.menu.idx = [a for _, a in g.menu.items].index("wing_save")
    g._menu_action(g.menu.handle("select"))
    pad_name = g.prompt.value

    class _Pad:
        name, layout = "DualSense Wireless Controller", "ps"

        def __init__(self):
            self.held = set()

        def pressed(self, n):
            return n in self.held

        def stick(self, which="left"):
            return (0.0, 0.0)

    g.pad = _Pad()
    g._pad_seeded, g._pad_prev = True, {}
    g.pad.held.add("cross")
    g._poll_pad(1 / 60)
    g.pad = None
    chk("a name another wing holds is never written over (saved beside it); the menu's 'Save the "
        "side wing as ...' opens the prompt on a free name and a pad's CROSS saves it",
        taken and rows.get("wing_save") == "Save the side wing as ..." and pad_name == "my side-3"
        and "my side-3" in lib.wings and not g.prompt.open and g.build.left.wing == "my side-3",
        f"{g.hint!r}")

    # --- SAVED WINGS -------------------------------------------------------------
    g.build.left.wing = "flank-e423"
    g.build.left.h = 0.90
    g.build.sync_mirror("left")
    g.sel = "left"
    keyd(pygame.K_l)
    wp = g.wings_page
    g.frame(1.0 / 60.0)
    order = list(wp.names)
    own = [n for n in order if not lib.wings[n].builtin]
    drawn = {n: (f_, w_) for n, f_, w_, _ in wp.drawn}
    agree = all(drawn[n] == wing_fit(lib.wings[n], "left", g.build, "corsa")[:2] for n in drawn)
    chk("L opens SAVED WINGS on the side slot: the player's own wings first, then the built-ins; "
        "each card drawn fits as wing_fit says (the 1.88 m wing and the top wings dimmed, their "
        "reasons in words)",
        g.page == "wings" and order[:len(own)] == sorted(own, key=str.lower)
        and all(lib.wings[n].builtin for n in order[len(own):])
        and agree and drawn["big side"][0] is False
        and drawn["rear-s1223"] == (False, "a top wing: select the top slot")
        and drawn["flank-e423"][0] is True and wp.current() == "flank-e423",
        f"{len(order)} cards, {len(drawn)} drawn")
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_saved_wings.png"))
    wp.set_names(wp.names, keep="big side")
    keyd(pygame.K_RETURN)
    refused = (g.page == "wings" and g.build.left.wing == "flank-e423"
               and "too wide for the Aurel Civetta: 1.88 m, the limit here is 1.50 m" in g.hint
               and "fits from h 1.09 m" in g.hint and "Unlimited" in g.hint)
    g.frame(1.0 / 60.0)
    shown_hint = list(g.saved_hint_drawn)
    wp.set_names(wp.names, keep="rear-s1223")
    keyd(pygame.K_RETURN)
    wrong = g.page == "wings" and "select the top slot" in g.hint
    keyd(pygame.K_3)
    top_now = g.sel == "top" and wing_fit(lib.wings["rear-s1223"], "top", g.build, "corsa")[0]
    keyd(pygame.K_RETURN)
    fitted_top = g.page == "car" and g.build.top.wing == "rear-s1223"
    chk("ENTER on a wing too wide for the slot fits nothing and says why and what would fit it "
        "(the height, Unlimited) on the page's bar; on a top wing with a side slot selected: "
        "the top slot; 3 selects it and ENTER fits it, back on the car page",
        refused and bool(shown_hint) and "too wide" in shown_hint[0] and wrong and top_now
        and fitted_top, f"{g.hint!r}")
    keyd(pygame.K_l)
    g.sel = "left"
    g.frame(1.0 / 60.0)
    i_my = wp.names.index("my side")
    rect = dict((i, r_) for i, r_ in wp._hits)[i_my]
    g._page_click(rect.center, False)
    sel1 = wp.current() == "my side" and g.page == "wings"
    g._page_click(rect.center, False)
    clicked = g.page == "car" and g.build.left.wing == g.build.right.wing == "my side"
    keyd(pygame.K_l)
    g.frame(1.0 / 60.0)
    chip = dict(wp.chips)["top"]
    g._page_click(chip.center, False)
    chip_ok = g.sel == "top"
    g.sel = "left"
    i0 = wp.idx
    keyd(pygame.K_RIGHT)
    keyd(pygame.K_DOWN)
    nav_ok = wp.idx == min(i0 + 1 + wp.cols, len(wp.names) - 1)
    i1 = wp.idx
    keyd(pygame.K_TAB)
    tab_ok = g.page == "cars"
    keyd(pygame.K_TAB)
    tab_ok = tab_ok and g.page == "wings"
    chk("SAVED WINGS takes the mouse: a click selects a card, a second click fits it (both sides, "
        "the car page); a slot chip selects the slot; RIGHT / DOWN a card / a row; TAB goes "
        "across to SAVED CARS and back",
        sel1 and clicked and chip_ok and nav_ok and tab_ok, f"idx {i0} -> {i1}")

    def tap(k, mod=0):
        keyd(k, mod)
        g._handle(pygame.event.Event(pygame.KEYUP, key=k, mod=mod))

    wp.set_names(wp.names, keep="plate")
    tap(pygame.K_DELETE)
    bi_del = "plate" in lib.wings and "cannot be deleted" in g.hint
    wp.set_names(wp.names, keep="carried side")
    tap(pygame.K_DELETE)
    armed = "carried side" in lib.wings and "DEL again" in g.hint
    tap(pygame.K_DELETE)
    gone = "carried side" not in lib.wings and "carried side" not in wp.names
    chk("SAVED WINGS: DEL twice deletes a wing of the player's (the first only arms it); a "
        "built-in is kept", bi_del and armed and gone, g.hint)
    #  more wings than a page holds: the bar says where the view is, and the
    #  wheel (anywhere on the page) scrolls it a row at a time
    extra = [f"spare {i}" for i in range(8)]
    for n_ in extra:
        lib.wings[n_] = mine.copy(name=n_)             # in memory: nothing to clean up
    wp.set_names(wp.names, keep=wp.names[0])
    g.frame(1.0 / 60.0)
    m0 = wp.more_text()
    top0 = wp.top
    for _ in range(wp.rows_vis + 1):
        g._page_wheel(-1)
    g.frame(1.0 / 60.0)
    m1 = wp.more_text()
    chk("SAVED WINGS with more wings than fit: the bar says which cards are in view, and the "
        "wheel scrolls a row at a time (the cursor with it)",
        m0.startswith("cards 1-12 of ") and m0.endswith("v  (wheel)") and top0 == 0
        and wp.top >= 1 and "^" in m1, f"{m0!r} -> {m1!r}")
    for n_ in extra:
        lib.wings.pop(n_, None)
    wp.refresh()

    # --- the diagram ------------------------------------------------------------
    surf = pygame.Surface((200, 140))
    box = (4, 4, 148, 132)
    surf.fill(C_HUD_BG)
    d_small = wing_diagram(surf, box, mine, g.text, 1.0)
    lit = pygame.surfarray.array3d(surf).astype(int)
    surf.fill(C_HUD_BG)
    d_big = wing_diagram(surf, box, big, g.text, 1.0)
    surf.fill(C_HUD_BG)
    d_car = wing_diagram(surf, box, carried, g.text, 1.0)
    surf.fill(C_HUD_BG)
    d_dim = wing_diagram(surf, box, mine, g.text, 1.0, dim=True)
    dim = pygame.surfarray.array3d(surf).astype(int)
    d_bus = wing_diagram(surf, box, huge, g.text, 1.0)
    orange = lambda a: int(((abs(a - np.array(C_PANEL_ON)) < 12).all(axis=2)).sum())  # noqa: E731
    chk("the diagram: one scale for spans up to DIAG_M_MAX (a 1.88 m wing drawn 2.35 x a 0.80 m one), a "
        "5 m one fitted to its card; plates drawn at the tips, two pylons on a pylon mount and "
        "none on an endplate one; dimmed it has none of the wing colour",
        abs(d_small["px_per_m"] - d_big["px_per_m"]) < 1e-9
        and abs(d_big["span_px"] / d_small["span_px"] - 1.88 / 0.80) < 1e-9
        and d_bus["span_px"] <= 148 - 12 + 1e-9 and d_bus["bar_m"] >= 1.0
        and d_small["plates"] == 2 and d_small["pylons"] == 2
        and d_car["pylons"] == 0 and d_car["plates"] == 2
        and d_small["plan"].bottom <= d_small["front"].top
        and d_small["front"].top < d_small["body_y"] <= d_small["front"].bottom
        and orange(lit) > 60 and orange(dim) == 0,
        f"{d_small['px_per_m']:.1f} px/m; wing-colour px {orange(lit)} lit, {orange(dim)} dimmed")
    t0 = time.perf_counter()
    for _ in range(6):
        g.frame(1.0 / 60.0)
    ms_w = (time.perf_counter() - t0) * 1e3 / 6
    chk("SAVED WINGS frame budget", *_budget(ms_w, 40.0))

    # --- SAVED CARS -----------------------------------------------------------------
    g.page = "car"
    keyd(pygame.K_g)
    cp_ = g.cars_page
    g.frame(1.0 / 60.0)
    first = cp_.rendered
    for _ in range(6):
        g.frame(1.0 / 60.0)
    pics = {n: cp_._pics[n][1] for n in cp_._pics}
    g.frame(1.0 / 60.0)
    cached = cp_.rendered == 0 and all(cp_._pics[n][1] is pics[n] for n in pics)
    from .prerace import pick_order
    chk("G opens SAVED CARS: every build as a card in the library's order (this car's first), "
        "a picture each, PICTURES_PER_FRAME a frame at most, then cached (the same surface "
        "the next frame, nothing drawn again)",
        g.page == "cars" and cp_.names == pick_order(lib.builds, "corsa")
        and cp_.names[0] == "my corsa" and 0 < first <= PICTURES_PER_FRAME
        and set(pics) == set(cp_.names) and cached,
        f"{cp_.names}, {first} the first frame")
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_saved_cars.png"))
    px = pygame.surfarray.array3d(pics["my corsa"]).astype(int)
    r_, g_, b_ = px[..., 0], px[..., 1], px[..., 2]
    body = int(((r_ - b_ > 50) & (g_ - b_ > 45) & (r_ - g_ < 40)).sum())     # the yellow paint
    wingpx = int(((r_ - g_ > 55) & (g_ - b_ > 25) & (r_ > 130)).sum())       # the wings' orange
    pic_ok = body > 1500 and wingpx > 60
    status = dict((n, s_) for n, s_, _ in cp_.drawn)
    chk("each picture is the car in its paint carrying its wings; each card says what ENTER "
        "does: load this car's, change to the Courier for its build",
        pic_ok and status["my corsa"] == "ENTER: load it"
        and status["my express (autosave)"] == "ENTER: change to the Courier and load it",
        f"{body} body px, {wingpx} wing px; {status}")
    #  save: the build's picture drawn afresh; rename: under its new name; delete: gone
    g.build = CarBuild.from_json(lib.builds["my corsa"])
    g.build.left.inc_deg += 3.0
    held = cp_._pics["my corsa"][1]
    g.save_build()
    for _ in range(3):
        g.frame(1.0 / 60.0)
    redrawn = cp_._pics["my corsa"][1] is not held
    cp_.set_names(cp_.names, keep="my corsa")
    keyd(pygame.K_r)
    typed("corsa wet")
    ren = ("corsa wet" in lib.builds and "my corsa" not in lib.builds
           and "my corsa" not in cp_._pics and cp_.current() == "corsa wet")
    for _ in range(3):
        g.frame(1.0 / 60.0)
    ren = ren and "corsa wet" in cp_._pics
    chk("SAVED CARS: a save draws the build's picture afresh, R renames it (the card and its "
        "picture under the new name)", redrawn and ren, g.hint)
    #  ENTER: this car's build loads; another car's: a bare garage says whose it is, the
    #  drive's garage asks for its car and build, a challenge's car refuses
    g.page = "cars"
    cp_.set_names(cp_.names, keep="corsa wet")
    keyd(pygame.K_RETURN)
    loaded = g.page == "car" and g.build.name == "corsa wet"
    gb = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, car="corsa")
    gb.open_saved("cars")
    gb.cars_page.set_names(gb.cars_page.names, keep="my express (autosave)")
    gb._page_activate()
    bare = (gb.car_wanted is None and gb._pending_action is None and gb.page == "cars"
            and "the Courier's build" in gb.hint)
    g.settings = _Settings46("real")
    g.open_saved("cars")
    cp_.set_names(cp_.names, keep="my express (autosave)")
    g.car_fixed = True
    keyd(pygame.K_RETURN)
    fixed = g._pending_action is None and "this challenge drives the Civetta" in g.hint
    g.car_fixed = False
    keyd(pygame.K_RETURN)
    act = g.frame(1.0 / 60.0)
    asked = (act == "car" and g.car_wanted == "express"
             and g.build_wanted == "my express (autosave)")
    chk("SAVED CARS ENTER: this car's build loads (the car page); another car's asks the drive "
        "for its car AND that build ('car' out of the frame); a bare garage and a challenge's "
        "car only say whose it is", loaded and bare and fixed and asked,
        f"{act} {g.car_wanted}/{g.build_wanted}; bare: {gb.hint!r}")
    g.car_wanted = g.build_wanted = None
    g.page = "cars"
    cp_.set_names(cp_.names, keep="my express (autosave)")
    keyd(pygame.K_d)
    d_other = g.settings.car_build == {"express": "my express (autosave)"} and "Courier" in g.hint
    cp_.set_names(cp_.names, keep="corsa wet")
    keyd(pygame.K_d)
    d_own = g.settings.build_of("corsa") == "corsa wet"
    g.frame(1.0 / 60.0)
    tags = [n for n, _s, _r in cp_.drawn]
    tap(pygame.K_DELETE)
    tap(pygame.K_DELETE)
    deleted = ("corsa wet" not in lib.builds and "corsa wet" not in cp_._pics
               and g.settings.build_of("corsa") == "" and "none now" in g.hint)
    chk("SAVED CARS: D makes the build ITS car's default (the Express's for an Express build); "
        "DEL twice deletes it, its picture and the default that named it",
        d_other and d_own and deleted and tags, g.hint)
    #  the drive's side: _garage_car opens the car on the chosen build (task 46), the
    #  menu's Change car (no build) as before
    import contextlib
    import io
    import sys
    from . import drive as _drv
    from types import SimpleNamespace as _NS
    #  (take_car_engine: task 48's Settings step, the car's own Engine)
    st = _NS(car="corsa", track="arena", car_build={}, build_of=lambda c=None: "", save=lambda: None,
             take_car_engine=lambda: False)
    hand = CarBuild.for_car("corsa")
    with contextlib.redirect_stdout(io.StringIO()):
        d_, seen_ = _drv._garage_car(sys.modules[__name__], lib, hand, "express", st,
                                     _NS(challenge=None), "corsa", build="my express (autosave)")
    st2 = _NS(car="corsa", track="arena", car_build={}, build_of=lambda c=None: "", save=lambda: None,
              take_car_engine=lambda: False)
    with contextlib.redirect_stdout(io.StringIO()):
        d2_, _s2 = _drv._garage_car(sys.modules[__name__], lib, hand, "express", st2,
                                    _NS(challenge=None, garage_hint=""), "corsa")
    chk("the drive's _garage_car opens the picked car ON the build the Saved cars page chose "
        "(and says so); without one, as the menu's Change car always did",
        seen_ == "express" and st.car == "express" and d_.name == "my express (autosave)"
        and d_.left.wing == "big side" and d2_.name != "my express (autosave)",
        f"{d_.name!r} / {d2_.name!r}")
    g.settings = None
    shutil.rmtree(root, ignore_errors=True)
    return ok


def _check_car_panel(g: "Garage", lib: Library, rep) -> bool:
    """Task 45's car page in a player's words, on a headless garage `g`: a
    fitted slot's panel opens on its summary (the drag in km/h, the weight
    in kg); no line the panel or the library page draws names the code
    (crossover., CONTRACT, V_REF, ESTIMATE, the study's); the station label
    reads the car's own axles; a downforce split never prints a share below
    0 % or over 100 %."""
    import re
    import cars as _cars
    ok = True

    def chk(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        rep(tag, passed, msg)

    xs = [0.05 * i for i in range(-140, 141)]            # -7 .. +7 m
    names = {"ahead of the front wheels", "at the front wheels", "between the wheels",
             "at the rear wheels", "behind the rear wheels"}
    seen = {station_label(x_, c) for c in _cars.CARS for x_ in xs}
    mx5, bus = station_label(-1.01, "rally"), station_label(1.5, "express")
    chk("the station label reads each car's own axles: the two-door rally car's x -1.01 has "
        "no door (its rear wheels), the Express's x 1.5 is ahead of its front wheels, five "
        "plain names on every car",
        "door" not in mx5 and mx5 == "at the rear wheels" and bus == "ahead of the front wheels"
        and seen == names,
        f"rally {mx5!r}; Express {bus!r}; Corsa x +0.97 {station_label(0.97)!r}")
    pct = re.compile(r"(-?\d+)%")
    bad, ends = [], set()
    for c in _cars.CARS:
        for x_ in xs:
            words, share = split_text(x_, car_spec(c))
            if not 0.0 <= share <= 1.0 or any(not 0 <= int(v) <= 100 for v in pct.findall(words)):
                bad.append((c, round(x_, 2), words))
            if "%" not in words:
                ends.add(words)
    xe = new_build("express").top.x
    chk("a downforce split never prints a share under 0 % or over 100 %: past an axle it "
        "says the load is all on that axle",
        not bad and len(ends) == 2, f"{bad[:3]}; Express top x {xe:+.2f}: "
        f"{split_text(xe, car_spec('express'))[0]!r}")
    code = ("crossover.", "CONTRACT", "V_REF", "ESTIMATE", "the study's")
    b0, sel0, vec0 = g.build.copy(), g.sel, g.view.show_vectors
    blit = g.text.blit
    try:
        g.page, g._armed = "car", None
        g.view.show_vectors = True                       # the legend too
        panels = {}
        for sk, wing in (("left", "flank-e423"), ("left", "fin"), ("top", "rear-s1223")):
            g.build.slot(sk).wing = wing
            g.sel = sk
            g.frame(1.0 / 60.0)
            panels[wing] = list(g.view.info_drawn) + [g.view.legend_drawn]
        heads = {w_: (ln[2], ln[3]) for w_, ln in panels.items()}
        words = [s_ for ln in panels.values() for s_ in ln]
        chk("a fitted slot's panel opens on what the wing does (drag in km/h, weight in "
            "kg); no panel line or arrow legend names the code",
            all("kg" in a_ and "drag" in a_ and "km/h" in b_ for a_, b_ in heads.values())
            and heads["flank-e423"][0].startswith("corner speed ")
            and heads["fin"][0].startswith("corner speed ")       # analysed or not
            and heads["rear-s1223"][0].startswith("downforce ")
            and not any(c_ in s_ for s_ in words for c_ in code)
            and all("km/h" in ln[-1] for ln in panels.values()),
            f"{[a_ for a_, _ in heads.values()]}; legend {panels['rear-s1223'][-1]!r}")
        drawn = []

        def spy(screen, s, *a, **kw):
            drawn.append(str(s))
            return blit(screen, s, *a, **kw)

        g.text.blit = spy
        g.lib_page.refresh()
        g.page, g.lib_page.focus = "library", "wings"
        names_ = [it[0] for it in g.lib_page.wings.items]
        g.lib_page.wings.idx = names_.index("fin")
        g.frame(1.0 / 60.0)
        notes = [s_ for s_ in drawn if s_.startswith("clean fin")]
        chk("the library page names no code: no '(CONTRACT s4)', the fin is a 'clean fin'",
            notes and not any(c_ in s_ for s_ in drawn for c_ in code),
            f"{notes[:1]}; {len(drawn)} strings")
    finally:
        del g.text.blit
        g.build, g.sel, g.view.show_vectors = b0, sel0, vec0
        g.lib_page.refresh()
        g.page, g._armed = "car", None
    return ok


def _check_airfoil_browse(g: "Garage", key, rep) -> bool:
    """Task 45: the airfoil library opened from the car page (A) with no
    wing being designed, on a headless garage `g` fresh on its car page.
    ENTER on a section, ENTER on the page's 'use' row, a double click on a
    section and one on that row all stay on the page and put 'design a wing
    first' in the hint -- which the page now draws -- where they used to do
    nothing in silence; the 'use' row is dimmed and the key bar promises no
    ENTER; the car is not touched; ESC goes back to it."""
    ok = True

    def chk(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        rep(tag, passed, msg)

    af, want = g.af_page, g.AF_NO_WING[False]
    fresh = not g._af_for_wing() and g.page == "car" and g.pad is None
    j0 = json.dumps(g.build.to_json(), sort_keys=True)
    idx0 = af.params.idx
    blit, drawn = g.text.blit, []

    def spy(screen, s, *a, **kw):
        drawn.append(str(s))
        return blit(screen, s, *a, **kw)

    def said(fn) -> str:
        g.hint = ""
        fn()
        return g.hint if g.page == "airfoil" else f"left for {g.page}"

    def dbl(w, i):
        _, ry, rh = next(h_ for h_ in w._hits if h_[0] == i)
        return lambda: g._page_click((w._rect.centerx, ry + rh // 2), False)

    try:
        key(pygame.K_a)
        use = next(p_ for p_ in af.params.params if p_.key == "use")
        af.focus = "list"
        on_list = said(lambda: key(pygame.K_RETURN))
        af.focus = "params"
        af.params.select_key("use")
        on_row = said(lambda: key(pygame.K_RETURN))
        g.text.blit = spy
        g.frame(1.0 / 60.0)                      # draws the hint, the bar, the rows' hits
        g.text.blit = blit
        #  the mouse: the section under the cursor clicked (it is already
        #  the selected one, so that is the second click of a double), then
        #  the dimmed row clicked with the cursor already on it
        by_click = said(dbl(af.list, af.list.idx))
        by_row = said(dbl(af.params, af.params.idx))
        heard = (on_list, on_row, by_click, by_row)
        chk("the airfoil library from the car page (no wing being designed): ENTER on a "
            "section or on 'use', and a double click on either, stay on the page and say "
            "'design a wing first' (drawn there); 'use' is dimmed, the bar promises no ENTER",
            fresh and all(h_ == want for h_ in heard) and not use.enabled
            and want in drawn and "use this section" not in drawn
            and "design a wing to use one" in drawn,
            f"{heard[0]!r}; {sum(h_ == want for h_ in heard)}/4 said it; 'use' enabled "
            f"{use.enabled}; drawn {want in drawn}")
        key(pygame.K_ESCAPE)
        chk("...the car untouched, and ESC goes back to it",
            json.dumps(g.build.to_json(), sort_keys=True) == j0 and g.page == "car", g.page)
    finally:
        g.text.__dict__.pop("blit", None)
        af.focus, af.params.idx = "list", idx0
        g.page, g.hint = "car", ""
    return ok


def _budget(ms: float, limit: float, calib=None):
    """`(passed, detail)` for a wall-clock frame-budget check, normalised by
    how slow the machine is RIGHT NOW.

    The three page-draw checks below are wall clock and they were false-alarm
    generators exactly as `render.py`'s V22 was: measured under a training run
    that had twelve cores busy, the car page came out at 15.39 ms against its
    14.0 ms budget and the module exited 1, which reads as a rendering
    regression and is not one. `render.frame_budget_verdict` already solves
    this by timing a fixed reference workload in the same process and scaling
    the budget by the slowdown; it is reused rather than reimplemented so the
    two cannot drift apart.

    `render` is NOT in `garage.py`'s import list in CONTRACT section 1 and this
    does not change that: the import is inside the SELF-CHECK, there is no
    cycle (`render` never imports `garage`), and nothing on the interactive or
    acceptance path reaches it. Noted in section 1.
    """
    from .render import frame_budget_verdict
    return frame_budget_verdict(ms, ms, budget_mean=limit, budget_p99=limit,
                                calib=calib)


def self_check(verbose: bool = True, screenshot_dir: str = "runs") -> bool:
    """Mesh sanity, closed-form parity with crossover.py, the build's two
    physics paths, every page headless, the DESIGN page on AeroBO's engine
    (its jobs replayed from the captured runs), the library."""
    import shutil
    import tempfile
    ok = True
    n_rows = [0, 0]                    # rows run, rows passed

    def rep(tag, passed, msg=""):
        nonlocal ok
        passed = bool(passed)
        ok = ok and passed
        n_rows[0] += 1
        n_rows[1] += int(passed)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag}: {msg}")

    mesh = build_car_mesh()
    rep("mesh", 100 <= len(mesh) <= 400, f"{len(mesh)} polygons")
    zs = np.concatenate([p[0][:, 2] for p in mesh])
    xs = np.concatenate([p[0][:, 0] for p in mesh])
    rep("mesh extents", abs(zs.max() - (CAR_H + 0.02)) < 1e-9 and zs.min() >= -0.03
        and abs(xs.max() - CAR_X_FRONT) < 1e-9 and abs(xs.min() - CAR_X_REAR) < 1e-9,
        f"z {zs.min():.2f}..{zs.max():.2f}, x {xs.min():.2f}..{xs.max():.2f}")
    inside = np.array([-0.15, 0.0, 0.70])
    bad = 0
    for verts, _, kind in mesh:
        if kind != "body":
            continue
        n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        if np.dot(n, verts.mean(axis=0) - inside) < 0:
            bad += 1
    rep("winding", bad == 0, f"{bad} inward-facing body polygons")

    d = WingDesign(wing="plate", x_w=0.97, h_w=0.90, inc_deg=0.0)
    g100 = d.gain_pct(100.0)
    k = 0.5 * RHO * 0.35 * 1.25
    ref = 100.0 * crossover.gain(k, 100.0, 0.97)
    rep("closed form == crossover.gain", abs(g100 - ref) < 1e-12,
        f"plate, front axle, R=100: {g100:+.4f}% (README: +2.1996%)")
    rep("README parity", abs(g100 - 2.1996) < 0.001, f"{g100:.4f} vs 2.1996")
    rep("force at the R=100 limit", abs(d.force_N() - 222.0) < 2.0,
        f"{d.force_N():.1f} N (README: 220 N sealed plate)")
    d2 = WingDesign(wing="fin", x_w=-1.90, h_w=0.9)
    d2.clamp()
    rep("clamp keeps the panel on the body", d2.x_w == X_W_MIN,
        f"x_w -1.90 -> {d2.x_w:+.3f} (min {X_W_MIN:+.3f})")
    d3 = WingDesign(wing="plate", inc_deg=15.0)
    rep("incidence raises CL to the stall clamp", abs(d3.cl_eff() - 1.6) < 1e-12,
        f"CL_eff {d3.cl_eff():.3f} at +15 deg (clamp {CL_STALL})")
    rep("cfg_kwargs round-trips", d.cfg_kwargs()["delta_dev_geom"] == 0.0
        and d.cfg_kwargs()["wing"] == "plate", str(d.cfg_kwargs()))

    tmp = tempfile.mkdtemp(prefix="carsim_garage_")
    lib = Library(os.path.join(tmp, "library"), use_xfoil=False)
    # --- the build: the published car maps onto the closed-form path exactly
    b = CarBuild.from_json(asdict(d))
    b.clamp(lib)
    kw = b.cfg_kwargs(lib)
    rep("published panel on both flanks -> closed-form VehicleConfig",
        kw == d.cfg_kwargs() and "dev_left" not in kw, str(kw))
    b.top.wing = "rear-s1223"
    kw2 = b.cfg_kwargs(lib)
    rep("adding a top wing -> designed path with the same published panel",
        kw2["wing"] == "off" and kw2["dev_left"] is not None and kw2["dev_left"].CL0 == 1.25
        and kw2["top"] is not None and kw2["top"].CZ > 0.3, f"top CZ {kw2['top'].CZ:.3f} CD {kw2['top'].CD:.4f}")
    b.left.wing = "flank-e423"
    b.sync_mirror("left")
    kw3 = b.cfg_kwargs(lib)
    rep("designed flank panel -> DevAero law", kw3["dev_left"].name == "flank-e423" and kw3["dev_right"].name == "flank-e423"
        and 0.3 < kw3["dev_left"].CL0 < 1.2, f"CL0 {kw3['dev_left'].CL0:.3f} CLa {kw3['dev_left'].CLa:.3f}")
    hud = b.hud_kwargs(lib)
    rep("hud kwargs", hud["dev_left"] and hud["top_on"] and hud["top_span"] > 1.0, str({k: hud[k] for k in ('dev_chord', 'top_span')}))
    # --- the wings' MASS, charged to the car (task 7) --------------------
    import cars as _cars
    mp = b.mass_points(lib)
    heavy = _cars.with_masses(_cars.CORSA_C, mp)
    empty = CarBuild()
    empty.clamp(lib)
    rep("three fitted wings are charged to the car at their own stations",
        len(mp) == 3 and all(0.5 < q.m < 20.0 for q in mp)
        and heavy.m > _cars.CORSA_C.m and heavy.h_cg > _cars.CORSA_C.h_cg
        and heavy.Izz > _cars.CORSA_C.Izz
        and _cars.with_masses(_cars.CORSA_C, empty.mass_points(lib)) is _cars.CORSA_C,
        f"{'+'.join(f'{q.m:.2f}' for q in mp)} = {heavy.m - _cars.CORSA_C.m:.2f} kg -> "
        f"m {heavy.m:.2f}  wdist_f {heavy.wdist_f:.4f}  h_cg {heavy.h_cg:.4f} "
        f"(+{1e3 * (heavy.h_cg - _cars.CORSA_C.h_cg):.1f} mm)  Izz {heavy.Izz:.1f}; "
        f"an EMPTY build leaves the stock car the same object")
    path = os.path.join(tmp, "design.json")
    b.save(path)
    back = CarBuild.load(path)
    rep("build json round-trip", back is not None and back.to_json() == b.to_json(), "")
    #  task 41: a build knows its car. The tag survives the file; a file from
    #  before (no "car") and a hand-edited non-string tag read as "" (any car);
    #  a new car's empty build is `CarBuild()` on its own slots, named for it
    tagged = b.copy()
    tagged.car = "rally"
    old_js = {k: v for k, v in b.to_json().items() if k != "car"}
    stock_new = {k: v for k, v in new_build("corsa").to_json().items() if k != "car"}
    rep("the build's car: round trip, old files any-car, a bad tag any-car; new builds "
        "named and tagged for their car",
        CarBuild.from_json(tagged.to_json()).car == "rally"
        and CarBuild.from_json(tagged.to_json()).to_json() == tagged.to_json()
        and CarBuild.from_json(old_js).car == "" and CarBuild.from_json(dict(old_js, car=7)).car == ""
        #  task 46: a build made for a retired car loads as a Corsa build
        and CarBuild.from_json(dict(old_js, car="mx5")).car == "corsa"
        and CarBuild.from_json(dict(old_js, car="bus")).car == "corsa"
        and tagged.to_json()["version"] == 2
        and default_build_name("corsa") == CarBuild().name == "my civetta"
        and default_build_name("rally") == "my halcón"
        and stock_new == {k: v for k, v in CarBuild().to_json().items() if k != "car"}
        and new_build("express").car == "express" and new_build("express").name == "my courier"
        and not new_build("express").left.wing
        and new_build("express").left.h > CarBuild().left.h,
        f"Express flank h {new_build('express').left.h:.2f} m")
    with open(path, "w") as f:
        json.dump(asdict(d), f)
    up = CarBuild.load(path)
    rep("a v1 WingDesign file upgrades to a build", up is not None and up.left.wing == "plate" and up.right.wing == "plate"
        and up.top.wing == "", str(up.left))
    rep("WingDesign.load refuses a v2 file", (b.save(path) and WingDesign.load(path) is None), "")
    got = []
    for body in ("[]", '{"version":2,"slots":[]}', '{"version":2,"slots":{"left":null}}',
                 '{"version":2,"slots":{"left":{"x":"abc"}}}', "{not json"):
        with open(path, "w") as f:
            f.write(body)
        got.append(type(CarBuild.load(path)).__name__)     # would raise before: garage disabled
    rep("a wrong-shaped design file loads as the empty car or None; an unreadable one is kept as .bad",
        got == ["CarBuild"] * 4 + ["NoneType"] and os.path.exists(path + ".bad")
        and not os.path.exists(path), str(got))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    g = Garage((1280, 800), b.copy(), headless=True, lib=lib)
    #  every file a check writes goes to the temp dir, never runs/ (the
    #  design rows below also switch AeroBO's own run records off)
    g.export_dir = os.path.join(tmp, "export")
    t0 = time.perf_counter()
    for i in range(12):
        g.deploy_cmd = 1.0 if i >= 4 else 0.0
        g.frame(1.0 / 60.0)
    ms = (time.perf_counter() - t0) * 1e3 / 12.0
    _ok, _why = _budget(ms, 14.0)
    rep("car page frame budget", _ok,
        f"{_why}, {g.view.n_polys} polys drawn (3 wings)")
    rep("deploy preview animates", 0.0 < g.deploy <= 1.0, f"deploy {g.deploy:.2f}")
    if screenshot_dir:
        os.makedirs(screenshot_dir, exist_ok=True)
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_frame.png"))
    #  the mount and the plates as the car page draws them (the owner,
    #  2026-09-25), with their pictures; the scene is redrawn below
    for tag_, ok_, msg_ in _wing_drawing_checks(lib, g.view, screenshot_dir):
        rep(tag_, ok_, msg_)
    # the player's paint (drive/paint.py) on the preview: the body is rebuilt
    # in it -- only when it changes -- and the scene drawn again at the same
    # pose changes the car's pixels, toward the new colour
    g.view.draw_scene(g.build, g.lib, g.cam, g.deploy, g.sel)
    f0 = pygame.surfarray.array3d(g.screen).astype(np.int32)
    body0 = g.view.car
    cobalt = (40, 72, 186)
    g.set_paint(cobalt)
    body1 = g.view.car
    g.set_paint(list(cobalt))                  # the same colour: no rebuild
    same = g.view.car is body1
    g.view.draw_scene(g.build, g.lib, g.cam, g.deploy, g.sel)
    f1 = pygame.surfarray.array3d(g.screen).astype(np.int32)
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_paint.png"))
    ch = (f0 != f1).any(axis=2)
    m0, m1 = (f_[ch].mean(axis=0) if ch.any() else np.zeros(3) for f_ in (f0, f1))
    g.set_paint(None)
    rep("paint: the preview body rebuilt in the paint (dark caps 0.75 of it) only on a "
        "change, drawn toward it; None is the stock yellow again",
        body1 is not body0 and same and g.view.car is not body1 and g.view.paint == C_PAINT
        and body0.colours.count(C_PAINT_DARK) == 2 and body1.colours.count((30, 54, 140)) == 2
        and body1.colours.count(cobalt) > 20 and C_PAINT not in body1.colours
        and int(ch.sum()) > 2000 and m0[0] > m0[2] and m1[2] > m1[0] + 40,
        f"{body1.colours.count(cobalt)} body polys in {cobalt}; {int(ch.sum())} px changed, "
        f"mean {m0.round().astype(int).tolist()} -> {m1.round().astype(int).tolist()}")

    def key(k, mod=0):
        return g._handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod))

    vecs = g.view.wing_vectors(g.build, g.lib, 1.0)
    kinds = {(k, kd) for k, _, _, _, kd in vecs}
    rep("wing force vectors: F and D per fitted wing at V_REF",
        all((k, kd) in kinds for k in SLOTS if g.build.slot(k).wing for kd in "FD"),
        f"{len(vecs)} arrows, {sorted(kinds)}")
    ok_dir = True
    for k, a, b, F, kd in vecs:
        d = b - a
        if kd == "D":
            ok_dir &= d[0] < 0 and abs(np.linalg.norm(d) - abs(F) * VEC_M_PER_N) < 1e-9
        elif k == "top":
            ok_dir &= d[2] < 0
        else:
            ok_dir &= (d[1] < 0) == (k == "left")      # into the car
    rep("vectors point into the car / down / rearward at ONE scale", ok_dir, f"{1 / VEC_M_PER_N:.0f} N = 1 m")
    #  task 45: the arrows start off (their labels piled over the wings on
    #  a first look); V shows them and V again hides them
    off0 = not g.view.show_vectors
    key(pygame.K_v)
    on1, hint1 = g.view.show_vectors, g.hint
    key(pygame.K_v)
    rep("the force arrows start off; V shows them, V again hides them",
        off0 and on1 and not g.view.show_vectors, f"{hint1!r}, then {g.hint!r}")

    x0 = g.build.left.x
    key(pygame.K_RIGHT)
    rep("RIGHT moves the selected flank forward (mirrored)", abs(g.build.left.x - x0 - 0.05) < 1e-9
        and abs(g.build.right.x - g.build.left.x) < 1e-9, f"x {g.build.left.x:+.3f}")
    key(pygame.K_3)
    key(pygame.K_UP)
    rep("3 selects the top slot; UP raises it", g.sel == "top" and g.build.top.h > 1.55, f"h {g.build.top.h:.2f}")
    key(pygame.K_1)
    key(pygame.K_w)
    rep("W cycles the slot's wing", g.build.left.wing != "flank-e423", f"left -> {g.build.left.wing!r}")
    key(pygame.K_w, pygame.KMOD_SHIFT)
    rep("SHIFT+W cycles back", g.build.left.wing == "flank-e423", g.build.left.wing)
    ok = _check_airfoil_browse(g, key, rep) and ok
    rep("ENTER -> drive", key(pygame.K_RETURN) == "drive", "")

    # --- task 41: the car the garage is given -------------------------------
    #  its slot bands: the stock cars share the garage's pre-41 bands
    #  (review of task 41, finding 4: the 540i clamps exactly as the Corsa),
    #  while a synthetic 12 m bus (a CarSpec-like stand-in: bodies reads its
    #  axles and its style from its name -- the retired Citaro's shell)
    #  keeps slots a Corsa would clamp
    from types import SimpleNamespace as _NS
    tall = _NS(name="test bus", L=5.845, wdist_f=1.0 - 3.741 / 5.845, t_f=2.110, t_r=2.110)

    def fitted(car, fx, fh, tx, th):
        cb = CarBuild()
        cb.left, cb.top = Slot("", fx, fh, 0.0), Slot("", tx, th, 6.0, "active")
        cb.clamp(lib, car)
        return (round(cb.left.x, 4), round(cb.left.h, 4), round(cb.top.x, 4), round(cb.top.h, 4))
    f_c, f_m = fitted("corsa", 1.65, 0.90, -0.90, 1.10), fitted("mx5", 1.65, 0.90, -0.90, 1.10)
    f_5 = fitted("540i", 1.65, 0.90, -0.90, 1.10)       # (task 46: 'mx5', a retired key,
    #                                                      is the Corsa itself)
    f_cb, f_b = fitted("corsa", 3.00, 1.60, -4.50, 3.30), fitted(tall, 3.00, 1.60, -4.50, 3.30)
    rep("CarBuild.clamp fits the car it is given: the 540i (and a retired car's key) clamp "
        "exactly as the Corsa (the garage's pre-41 bands), a bus keeps its 1.60 m flank and "
        "3.30 m top wing (a Corsa clamps all four)",
        f_c == (round(CAR_X_FRONT - 0.5 * DEV_CHORD, 4), 0.9, -0.9, round(deck_z(-0.9) + 0.14, 4))
        and f_m == f_c and f_5 == f_c and f_b == (3.0, 1.6, -4.5, 3.3)
        and f_cb[1] == H_W_MAX and f_cb[3] == TOP_H_MAX,
        f"corsa {f_c}  mx5 {f_m}  540i {f_5}  bus {f_b}")
    #  bodies' copy of this garage's Corsa deck is these STATIONS' numbers
    rep("bodies.LEGACY_DECK is this garage's STATIONS deck (x, z_top), and the band rule reads it",
        bodies.LEGACY_DECK == tuple((st[0], st[3]) for st in STATIONS)
        and all(bodies.legacy_deck_z(x_) == deck_z(x_) for x_ in np.linspace(-2.2, 1.9, 83))
        and all(top_h_band(k_, x_)[0] == deck_z(x_) + TOP_STOW_GAP + 0.08
                for k_ in _cars.STOCK_CARS for x_ in (TOP_X_MIN, -1.62, -0.9, TOP_X_MAX)),
        f"{len(STATIONS)} stations")
    #  a build saved before task 41 at the old band's extremes -- both ends of
    #  every band -- loads on the stock cars with nothing moved: the
    #  same build_id, so its PBs stay attached (finding 4's scenario)
    from .records import build_id as _bid
    ext = []
    #  (task 46: the rear end is where the top wing's pylons still stand on
    #  the car -- the old band's floor put a 0.30 m wing's feet behind the tail)
    tx0 = max([TOP_X_MIN] + [top_x_floor(k_, lib.wings["rear-s1223"]) for k_ in ("corsa", "mx5", "540i")])
    for fx_end, fh, tx, th in ((0, H_W_MIN, tx0, None), (1, H_W_MAX, TOP_X_MAX, TOP_H_MAX)):
        c_ = lib.wings["flank-e423"].chord
        fx = (CAR_X_REAR + 0.5 * c_) if fx_end == 0 else (CAR_X_FRONT - 0.5 * c_)
        th = deck_z(tx) + TOP_STOW_GAP + 0.08 if th is None else th
        ext.append(dict(version=2, name=f"pre41 end {fx_end}", mirror=True, builtin=False,
                        slots=dict(left=dict(wing="flank-e423", x=fx, h=fh, inc_deg=0.0,
                                             mode="active"),
                                   right=dict(wing="flank-e423", x=fx, h=fh, inc_deg=0.0,
                                              mode="active"),
                                   top=dict(wing="rear-s1223", x=tx, h=th, inc_deg=6.0,
                                            mode="active"))))
    moved = [(e["name"], k_) for e in ext for k_ in _cars.STOCK_CARS
             if _bid("", CarBuild.from_json(e).clamp(lib, k_).to_json()) != _bid("", e)]
    rep("a pre-41 build at the old extremes loads on the stock cars unmoved",
        not moved, f"top x {TOP_X_MIN:+.4f} h {ext[0]['slots']['top']['h']:.4f} / "
        f"x {TOP_X_MAX} h {TOP_H_MAX}, flank x {ext[0]['slots']['left']['x']:+.4f} / "
        f"{ext[1]['slots']['left']['x']:+.4f}: same build_id" if not moved else str(moved))
    rb = CarBuild.for_car(tall)
    rep("a fresh car and R's reset put the slots at the car's own defaults",
        (rb.left.x, rb.left.h, rb.top.x, rb.top.h) == (3.00, 1.60, -4.50, 3.30)
        and CarBuild.for_car("540i").to_json() == CarBuild().to_json()
        and CarBuild.for_car("mx5").to_json() == CarBuild().to_json(),
        f"bus flank ({rb.left.x}, {rb.left.h}) top ({rb.top.x}, {rb.top.h})")
    #  the preview is that car's body; the Corsa's mesh is the one above, bit for bit
    bus_mesh = build_car_mesh(car="rally")
    bb_ = bodies.body("rally")
    zs_b = np.concatenate([q[0][:, 2] for q in bus_mesh])
    xs_b = np.concatenate([q[0][:, 0] for q in bus_mesh])
    in_b = np.array([0.5 * (bb_.x_front + bb_.x_rear), 0.0, 0.5 * (bb_.ground + bb_.height)])
    bad_b = sum(1 for v_, _c, k_ in bus_mesh if k_ == "body"
                and np.dot(np.cross(v_[1] - v_[0], v_[2] - v_[0]), v_.mean(axis=0) - in_b) < 0)
    same_c = all(np.array_equal(a_[0], b_[0]) and a_[1:] == b_[1:]
                 for a_, b_ in zip(build_car_mesh(), build_car_mesh(car="corsa")))
    rep("the preview draws the fitted car: the rally Escort lofted from its own shell, "
        "outward, its livery on the top",
        same_c and bad_b == 0 and abs(zs_b.max() - (bb_.height + 0.02)) < 1e-9
        and abs(xs_b.max() - bb_.x_front) < 1e-9 and abs(xs_b.min() - bb_.x_rear) < 1e-9
        and sum(1 for _v, c_, _k in bus_mesh if c_ == C_LIVERY) == 10,
        f"{len(bus_mesh)} polys, x {xs_b.min():.2f}..{xs_b.max():.2f}, z to {zs_b.max():.2f}")
    #  the review of task 41's root design: an any-car (or another car's)
    #  build only LOOKED at on another car is not moved -- the garage edits a
    #  fitted copy and hands the original back; an edit hands the copy back;
    #  this car's own build is edited in place
    anyb = CarBuild(name="any old")
    anyb.left.wing, anyb.top.wing, anyb.top.h = "flank-e423", "rear-s1223", 1.60
    anyb.sync_mirror("left")
    any_js = anyb.to_json()
    gv = Garage((1280, 800), anyb, headless=True, lib=lib, car="express")
    looked = (gv.build is not anyb and gv.build.top.h > 1.85 and gv.handed_back() is anyb
              and anyb.to_json() == any_js)
    gv._move(dinc=1.0)
    edited = gv.handed_back() is gv.build
    lib.save_build(any_js)
    gv._load_build("any old")
    reloaded = gv.handed_back().to_json() == any_js and gv.build.top.h > 1.85
    ownb = new_build("express")
    go = Garage((1280, 800), ownb, headless=True, lib=lib, car="express")
    rep("a build of another car / any car is viewed as a fitted copy and handed back unmoved; "
        "an edit hands the copy back; this car's own build is edited in place",
        looked and edited and reloaded and go.build is ownb and go.handed_back() is ownb,
        f"any-car top h 1.60 shown at {gv.build.top.h:.2f} on the Express")
    lib.delete("builds", "any old")
    #  task 45: a first launch's car -- the Corsa's defaults, 'my corsa', any
    #  car (drive.py's launch with no garage_design.json) -- opened on the
    #  Express is called 'my express': the BUILD line, the menu's save row,
    #  S's prompt. Looked at only, it goes back as it came; an edit hands back
    #  'my express'. This car's own build so called is renamed in place; a
    #  name the player chose, and the library's own 'my corsa' loaded, stay
    first = CarBuild.from_json(dict(wing="off", x_w=0.97, h_w=0.90, inc_deg=0.0))
    first_js = first.to_json()
    ge = Garage((1280, 800), first, headless=True, lib=lib, car="express")
    head = "".join(t_ for t_, _ in ge.build_header())
    ge._menu_open()
    row = next(lbl for lbl, a in ge.menu.items if a == "build_save")
    ge.menu.hide()
    ge.prompt_build()
    asked = ge.prompt.value
    ge.prompt.open_ = False
    looked = ge.handed_back() is first and first.to_json() == first_js and first.name == "my civetta"
    ge._move(dinc=1.0)
    edited = ge.handed_back().name == "my courier"
    own_e = CarBuild(name="my civetta", car="express")
    in_place = Garage((1280, 800), own_e, headless=True, lib=lib, car="express").build is own_e
    chosen = Garage((1280, 800), CarBuild(name="Fast"), headless=True, lib=lib, car="express")
    corsa_b = CarBuild(name="my civetta", car="corsa")
    lib.save_build(corsa_b.to_json())
    gl_ = Garage((1280, 800), CarBuild(name="Fast"), headless=True, lib=lib, car="express")
    gl_._load_build("my civetta")
    lib.delete("builds", "my civetta")
    rep("a first launch's 'my civetta' on the Courier is 'my courier' in the garage (BUILD line, "
        "save row, S's prompt), goes back as it came when only looked at, 'my courier' once "
        "edited; the Courier's own 'my civetta' is renamed in place; a chosen name and the "
        "library's 'my civetta' stay",
        "BUILD  my courier" in head and "'my courier'" in row and asked == "my courier"
        and looked and edited and in_place and own_e.name == "my courier"
        and chosen.build.name == "Fast" and gl_.build.name == "my civetta"
        and not other_car_name(new_build("rally"), "rally", lib),
        f"'{head}' / '{row}' / prompt '{asked}'")
    real, unl = _NS(wing_limits="real"), _NS(wing_limits="unlimited")
    wide = lib.wings["flank-e423"].copy(name="tall-fin", builtin=False)
    wide.span = 1.60                        # past a Corsa flank's 1.50 m limit at h 0.90
    lib.save_wing(wide)
    gb = Garage((1280, 800), CarBuild.for_car("express"), headless=True, lib=lib, car="express",
                settings=real)
    gb.frame(1.0 / 60.0)
    rep("the orbit camera stands back for the tall Express (the Corsa's view unchanged)",
        abs(gb.cam.dist - 7.6 * gb.view.geo.scale) < 1e-9 and gb.view.geo.scale > 1.2
        and g.cam.scale == 1.0 and gb.car == "express", f"scale {gb.view.geo.scale:.2f}, "
        f"distance {gb.cam.dist:.1f} m")
    #  REAL: W skips a wing past this slot's limit and says why; DOWN stops
    #  where the panel's lower tip reaches the ground clearance
    gr = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, settings=real)
    seen = set()
    for _ in range(12):
        gr._cycle_wing(1)
        seen.add(gr.build.left.wing)
    gr._cycle_wing(1)
    while gr.build.left.wing != "flank-e423":
        gr._cycle_wing(1)
    hint_w = ""
    for _ in range(8):
        gr._cycle_wing(1)
        hint_w = gr.hint if "skipped" in gr.hint else hint_w
    rep("Real: W never fits a wing past this slot's span limit, and says why and where",
        "tall-fin" not in seen and "flank-e423" in seen and "1.50 m span limit" in hint_w
        and "Unlimited" in hint_w, hint_w)
    #  task 45: W names a built-in the way a player reads it, with where it
    #  is in the cycle and what it does -- never by its library key; 'no
    #  wing' is the cycle's last step; the player's own wing keeps its name;
    #  the library's rows and the build's one-line summary say the same, and
    #  the rows tag a built-in 'built in'
    gn = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, settings=unl)
    order = [""] + sorted(n_ for n_, w_ in lib.wings.items() if w_.role == "flank")
    said = {}
    for _ in range(len(order)):
        gn._cycle_wing(1)
        said[gn.build.left.wing] = gn.hint
    gn.sel = "top"
    gn._cycle_wing(1)
    top_said = gn.hint
    m_ = len(order)
    named = all(said[k].startswith(f"left: {BUILTIN_WING_SHOWN[k][0]} ({order.index(k)} of {m_}) - "
                                   f"{BUILTIN_WING_SHOWN[k][1]}")
                for k in ("fin", "plate", "flank-e423"))
    gn.lib_page.refresh()
    rows = {it[0]: it for it in gn.lib_page.wings.items}
    rep("W names a built-in wing as a player reads it, '(n of m)' and what it does, never "
        "its key; 'no wing' is the last step; the library's rows and the build summary name it the same",
        named and said[""].startswith(f"left: no wing ({m_} of {m_})")
        and said["tall-fin"].startswith(f"left: tall-fin ({order.index('tall-fin')} of {m_})")
        and not any(k in s_ for s_ in [*said.values(), top_said] for k in ("flank-e423", "rear-s1223"))
        and top_said.startswith("top: Rear wing (1 of ")
        and rows["plate"][3] == "Side plate" and rows["plate"][2] == "flank, built in"
        and rows["plate"][1].startswith(BUILTIN_WING_SHOWN["plate"][1])
        and rows["tall-fin"][3] == "tall-fin" and "built in" not in rows["tall-fin"][2]
        and "top: Rear wing x " in gn.build.summary(lib),
        f"{said['flank-e423']!r}; {said['']!r}; {top_said!r}")
    #  round 3: the TIME TRIAL page's 'Try ready-made wings' opens the garage
    #  with W pressed on the first empty slot (the mirror fits both flanks);
    #  a car with every slot filled is left alone
    gt = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, settings=real)
    gt.page = "library"
    tried = gt.try_ready_made()
    full = CarBuild.for_car("corsa")
    full.left.wing, full.top.wing = "plate", "rear-s1223"
    full.sync_mirror("left")
    gf = Garage((1280, 800), full, headless=True, lib=lib, settings=real)
    rep("'Try ready-made wings': the car page, the first empty slot, W pressed once "
        "(a ready-made wing on both flanks), the hint says what next; a full car is left alone",
        tried and gt.page == "car" and gt.sel == "left"
        and gt.build.left.wing == order[1] == gt.build.right.wing and not gt.build.top.wing
        and gt.hint.startswith(f"left: {wing_shown(order[1], lib)[0]} (1 of ")
        and gt.hint.endswith("W: the next one, ENTER: drive it")
        and not gf.try_ready_made() and gf.build.left.wing == "plate" and gf.hint == "",
        gt.hint)
    guide = gt.hint
    gt._bg_hint("wing data ready: e423")
    kept = gt.hint == guide
    gt._t_alive += HINT_S + 0.1
    gt._bg_hint("wing data ready: e423")
    rep("... and the wing data arriving does not cover that hint while it shows, only after",
        kept and gt.hint == "wing data ready: e423", gt.hint)
    while gr.build.left.wing != "flank-e423":
        gr._cycle_wing(1)
    gr.build.left.h = 0.60
    gr.build.sync_mirror("left")
    gr.sel = "left"
    h_seen = []
    for _ in range(4):
        gr._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
        h_seen.append(round(gr.build.left.h, 4))
    floor = 0.15 + 0.5 * lib.wings["flank-e423"].span
    rep("Real: DOWN stops with the panel's lower tip at the ground clearance",
        h_seen == [round(floor, 4)] * 4 and "ground clearance" in gr.hint
        and bodies.over_limits(gr.build, lib, "corsa") == [],
        f"h 0.60 -> {h_seen} (floor {floor:.2f} m = 0.15 + {lib.wings['flank-e423'].span:.2f}/2)")
    #  UNLIMITED (task 45): DOWN goes past the clearance but stops with the
    #  lower tip at the road, never below it; the move that crosses the limit
    #  says the runs stop being official (once: the stop's hint follows), and
    #  UP back over it says so too
    gu = Garage((1280, 800), gr.build.copy(), headless=True, lib=lib, settings=unl)
    gu.sel = "left"
    span_e = lib.wings["flank-e423"].span
    tips_u, hints_u = [], []
    for _ in range(12):
        gu._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
        tips_u.append(gu.build.left.h - 0.5 * span_e)
        hints_u.append(gu.hint)
    ups_u = []
    for _ in range(6):
        gu._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
        ups_u.append(gu.hint)
    rep("Unlimited: DOWN goes past the clearance, stops with the lower tip at the road (0 m); "
        "crossing the limit says the runs count as UNLIMITED, and back says so",
        tips_u[0] < 0.15 - 1e-9 and min(tips_u) > -1e-9 and abs(tips_u[-1]) < 1e-9
        and "left is now past the Civetta's" in hints_u[0] and "UNLIMITED (not official)" in hints_u[0]
        and "at the road" in hints_u[-1]
        and any(s_.startswith("left is within the Civetta's limit again") for s_ in ups_u),
        f"tips {tips_u[0]:+.2f} .. {tips_u[-1]:+.2f} m; '{hints_u[0]}'")
    #  a panel too long for the band stays up at its stop; one saved BELOW
    #  the road is left where it is (Unlimited)
    stops_u = []
    for h_start in (1.20, 0.40):
        gt = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, settings=unl)
        gt.build.left.wing, gt.build.left.h, gt.sel = "tall-fin", h_start, "left"
        gt.build.sync_mirror("left")
        for _ in range(20):
            gt._move(dh=-STEP_H)
        stops_u.append(round(gt.build.left.h, 4))
    rep("Unlimited: a 1.60 m panel stops at h 0.80 (its tip on the road); one saved at h 0.40 "
        "(tip 0.40 m under it) is not moved",
        stops_u == [0.8, 0.4], f"from 1.20 -> {stops_u[0]:.2f}, from 0.40 -> {stops_u[1]:.2f}")
    names_u, w_ok, w_in = set(), True, 0
    gu.build.left.h = 0.90
    gu.build.sync_mirror("left")
    for _ in range(12):
        was_u = "left" in gu._limits_past()
        gu._cycle_wing(1)
        now_u = "left" in gu._limits_past()
        names_u.add(gu.build.left.wing)
        w_ok = (w_ok and ("UNLIMITED (not official)" in gu.hint) == (now_u and not was_u)
                and ("limit again" in gu.hint) == (was_u and not now_u))
        w_in += now_u and not was_u
    rep("Unlimited: W reaches every wing, and the wing that takes the slot past its limit "
        "(or back) says so",
        "tall-fin" in names_u and w_ok and w_in >= 1,
        f"wings {sorted(n for n in names_u if n)}, {w_in} crossing(s) said")
    #  a build already past the limit, loaded in Real mode, is KEPT, and the
    #  page says its runs count as Unlimited
    bp = CarBuild.for_car("corsa")
    bp.left.wing = "tall-fin"
    bp.sync_mirror("left")
    gp = Garage((1280, 800), bp, headless=True, lib=lib, settings=real)
    gp.frame(1.0 / 60.0)
    rows_p = limit_rows(gp.build, lib, "corsa")
    h0 = gp.build.left.h
    gp.sel = "left"
    gp._move(dh=-STEP_H)
    rep("Real: a build already past its limit is kept, listed PAST THE LIMIT, not moved lower",
        gp.build.left.wing == "tall-fin" and [r_[0] for r_ in rows_p if r_[3]] == ["left", "right"]
        and abs(rows_p[0][2] - 1.50) < 1e-9 and gp.build.left.h == h0,
        f"{rows_p[0]}")
    #  task 45: LOADING it says so too (the top slot selected: a flank that
    #  crossed is still named); the SPAN LIMITS head gives up "limits: "
    #  before any car's name is cut
    bp.name, bp.car = "wide load", "corsa"
    lib.save_build(bp.to_json())
    gl = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, settings=real)
    gl.sel = "top"
    gl._load_build("wide load")
    hint_l = gl.hint
    lib.delete("builds", "wide load")
    heads = {}
    for ck in _cars.CAR_ORDER:
        for st_ in (real, unl):
            bh = CarBuild.for_car(ck)
            bh.left.wing = "tall-fin"
            bh.sync_mirror("left")
            gh = Garage((1280, 800), bh, headless=True, lib=lib, car=ck, settings=st_)
            gh.frame(1.0 / 60.0)
            heads[ck, st_.wing_limits] = gh.view.limits_drawn
    cut = [k_ for k_, (h_, _m) in heads.items() if h_.endswith("...") or not h_]
    rep("loading a build past the limit says its runs count as UNLIMITED; SPAN LIMITS never "
        "cuts a car's name ('limits: Unlimited' -> 'Unlimited' first)",
        "left is now past the Civetta's 1.50 m limit" in hint_l and "UNLIMITED" in hint_l
        and not cut and heads["corsa", "unlimited"] == ("SPAN LIMITS  Aurel Civetta 1.2", "Unlimited"),
        f"'{hint_l}'; cut {cut}; Corsa {heads['corsa', 'unlimited']}")
    #  THE DESIGNER on the car's limits (task 41 wired into AeroBO's designer):
    #  each slot's session reads the car it is fitted to (`drive/bodies.py`
    #  through `aerobo_bridge`): the Express's flank span row opens at ITS
    #  limit (the lower tip at its own 0.16 m underbody, at the slot's h), its
    #  top wing rides over its own deck, up to 0.41 m over its roof, and is
    #  1.2 x its width at most; a Corsa flank moved down to h 0.40 caps at
    #  0.50 m. (Task 46: the Express in the retired bus's role.)
    #  (The old carsim designer's commit / re-cap rows are the AeroBO
    #  designer's in its own section below: S refuses a wing past the limit.)
    sb_f, sb_t = gb.design_page.session_for("left"), gb.design_page.session_for("top")
    bb = bodies.body("express")
    of_b, ot_b = sb_f.op, sb_t.op
    hb = gb.build.left.h
    t_lo, t_hi = bodies.top_h_band("express", gb.build.top.x)
    gl = Garage((1280, 800), CarBuild.for_car("corsa"), headless=True, lib=lib, settings=real)
    gl.build.left.h = H_W_MIN
    gl.build.sync_mirror("left")
    ol = gl.design_page.session_for("left").op
    rep("the WingLab designer reads the car: the Express's flank span row is its own limit at "
        "h, its top wing rides its own band over its own deck; a Corsa flank at h 0.40 caps at "
        "0.50 m",
        of_b.car == "express" and abs(of_b.limit - 2.0 * (hb - bb.ground)) < 1e-12
        and abs(of_b.size_rows["b_m"][1] - of_b.limit) < 1e-6
        and abs(ot_b.deck - bb.deck_z(gb.build.top.x)) < 1e-12
        and (ot_b.ride_band[0], ot_b.ride_band[1]) == (t_lo, t_hi)
        and abs(ot_b.size_rows["b_m"][1] - 1.2 * bb.width) < 1e-6
        and abs(ol.limit - 0.50) < 1e-12 and abs(ol.size_rows["b_m"][1] - 0.50) < 1e-6,
        f"Express flank <= {of_b.limit:.2f} m at h {hb:.2f} (rows b {of_b.size_rows['b_m']}), top "
        f"deck {ot_b.deck:.2f} ride {ot_b.ride_band[0]:.2f}-{ot_b.ride_band[1]:.2f} m, b <= "
        f"{ot_b.size_rows['b_m'][1]:.3f} m; Corsa h 0.40: <= {ol.limit:.2f} m")
    #  a Param's band may be a callable, read at every step (the span row's)
    pc = ui.Param("t", "t", lambda: 1.0, lambda v: None, lo=0.0, hi=lambda: 1.25, step=0.5)
    got_pc = []
    pc.set = got_pc.append
    pc.adjust(+1)
    rep("a row's band may move: a callable hi is read at the step", got_pc == [1.25]
        and pc.band() == (0.0, 1.25), str(got_pc))

    # --- the designer: AeroBO's own engine (PLAN2 §9.4) -----------------------
    #  Every design job below REPLAYS a captured AeroBO run
    #  (drive/data/aerobo_fixtures, `aerobo_models.use_fixtures`) through the
    #  same worker-thread EngineJob path a real one takes; the law and the
    #  design report are AeroBO's evaluator for real (no XFOIL). XFOIL is
    #  switched off for the check so the sections sit at AeroBO's library
    #  point on every machine (PLAN2 D11), and nothing is written under runs/.
    am = _am()
    api = am.bridge.api
    rp = am.use_fixtures()
    env_nx = os.environ.get("CARSIM_NO_XFOIL")
    os.environ["CARSIM_NO_XFOIL"] = "1"
    g.aerobo_dir = None
    try:
        key(pygame.K_d)
        #  STEP 1. D opens the MISSION, which gates everything downstream.
        rep("D opens the MISSION for the selected slot (step 1 of 2)",
            g.page == "mission" and g.sel == "left" and not g.mission.stated,
            f"{g.page}, circuit {g.mission.track}")
        t0 = time.perf_counter()
        g.frame(1.0 / 60.0)
        g.frame(1.0 / 60.0)
        mspf = (time.perf_counter() - t0) * 1e3 / 2
        rep("mission page draws", *_budget(mspf, 60.0))
        if screenshot_dir:
            pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_mission.png"))
        mp = g.mission_page
        #  the JOBS (the owner, 2026-09-25: "One more circuit should be added
        #  'Stopping'. Left flank and right flank, should be side."; 2026-09-26,
        #  "why no longer circuits?", "Side wing should also have 'stopping'
        #  mission"): the side wing flies a job of its own, a circuit or the
        #  stop on the air brake; the top wing's circuit lap and its stop both
        #  fly against the bare car
        from .aero import mission as _msn     # `ms` is a local number here
        side_ok = (mp.job() == g.mission.side_track and mp.flies()
                   and isinstance(mp.result, _msn.LapResult) and mp.result.ok
                   and mp.job_choices() == list(_msn.SIDE_JOBS))
        sig_side = mp.signature()
        g.mission.side_track = _msn.STOPPING
        mp.update()
        air = mp.result
        side_stop_ok = (mp.job() == _msn.STOPPING and isinstance(air, _msn.StopResult)
                        and air.ok and mp.bare.ok and mp.flies()
                        and mp.signature()[0] == "stopping 100 km/h"
                        and mp.params.param("vstop").enabled)
        g.mission.side_track = "arena"
        mp.update()
        g.sel = "top"
        mp.update()
        lap_ok = (mp.job() == g.mission.track and isinstance(mp.result, _msn.LapResult)
                  and mp.result.ok and mp.bare is not None and mp.bare.ok
                  and mp.job_choices() == list(_msn.JOBS))
        lap_txt = (f"{mp.result.time:.3f} s vs {mp.bare.time:.3f} s bare, "
                   f"{len(mp.profile().corners)} corners" if lap_ok else mp.err)
        g.mission.track = _msn.STOPPING
        mp.update()
        stop_ok = (mp.job() == _msn.STOPPING and isinstance(mp.result, _msn.StopResult)
                   and mp.result.ok and mp.bare.ok and mp.flies()
                   and mp.signature()[0] == "stopping 100 km/h")
        stop_txt = (f"stop {mp.result.distance:.2f} m vs {mp.bare.distance:.2f} m bare"
                    if stop_ok else mp.err)
        side_stop_ok = side_stop_ok and air.distance <= mp.result.distance
        g.mission.track = "linden"
        g.sel = "left"
        side_kept = mp.signature() == sig_side
        g.mission.track = "arena"
        mp.update()
        rep("the jobs: a side wing flies its own job (a circuit, or the stop on the air brake; "
            "the top wing's job does not move it); the top wing's circuit lap and its Stopping "
            "job both fly against the bare car",
            side_ok and side_kept and side_stop_ok and lap_ok and stop_ok,
            f"{lap_txt}; {stop_txt}; "
            + (f"air brake {air.distance:.2f} m" if side_stop_ok else "the side stop FAILED"))

        #  the GATE, which is the point of the mission being first
        g.open_designer("left")
        rep("the design page is GATED on a stated mission", g.page == "mission", g.hint)

        #  STEP 2: the DESIGN navigator, on the slot's AeroBO session.
        key(pygame.K_RETURN)
        dp = g.design_page
        rep("ENTER states the mission and opens the DESIGN navigator",
            g.page == "section" and g.mission.stated and dp.wing is not None,
            f"{g.page}, at '{dp.nav.current()}'")
        rep("the navigator carries the whole procedure, in order",
            [k for _, st in DESIGN_TREE for k, _ in st] == dp.nav.keys and len(dp.nav.keys) == 16,
            " > ".join(gname.split("  ")[0] for gname, _ in DESIGN_TREE))
        #  every one of the sixteen views draws on a fresh session (their
        #  empty states), each checked to be in the tree -- drawn and
        #  hit-recorded -- right after its own select
        worst, worst_k, unseen = 0.0, "", []
        for node in dp.nav.keys:
            dp.nav.select(node, force=True)
            t0 = time.perf_counter()
            g._draw_page()
            dt = (time.perf_counter() - t0) * 1e3
            if dt > worst:
                worst, worst_k = dt, node
            if node not in [k for k, _, _ in dp.nav._hits]:
                unseen.append(node)
        rep("all 16 navigator panels draw", not unseen and not g.shell.view_errors,
            f"worst {worst:.1f} ms at '{worst_k}'; every step in the tree once selected"
            + (f" -- NOT drawn: {unseen}" if unseen else "")
            + (f" -- render errors: {sorted(g.shell.view_errors)}" if g.shell.view_errors else ""))

        #  THE GATE is AeroBO's (PLAN2 D10): once the mission is stated, 2
        #  Airfoil, 2.8 Endplate and 3 Wing are all open -- the wing flies
        #  the family's own sections until one is chosen -- and 4 Results
        #  waits for a completed run, with the reason.
        dp.open("left")
        s, af, ep, w = dp.session, dp.af, dp.ep, dp.wing
        open_now = {k: dp.nav.open(k) for k in dp.nav.keys}
        rep("a fresh page opens on 2 Airfoil; 2, 2.8 and 3 are open (WingLab's gate: the wing flies "
            "the family's own sections until one is chosen), 4 Results is shut",
            dp.nav.current() == "af.screen"
            and all(v for k, v in open_now.items() if not k.startswith("r."))
            and not any(v for k, v in open_now.items() if k.startswith("r.")),
            f"{sum(open_now.values())} of {len(open_now)} open")
        rep("4 Results is REFUSED before a run, with WingLab's reason",
            dp.nav.select("r.summary") is False and dp.nav.current() == "af.screen"
            and "no completed run" in dp.nav.refused, dp.nav.refused[:64])

        def view(k_):
            """Show view `k_` (the page's own book-keeping, not the shell's
            select: the shell has its own check)."""
            dp.nav.select(k_, force=True)
            g.page = "section"

        # -- 1 Mission: AeroBO's operating point and search policy ------------
        of = s.op
        v_side = am.bridge._lap_v(g.build, g.mission.for_side(), g.lib).v_mean
        rep("the slot's session flies WingLab's operating point: V is the side wing's circuit's "
            "lap mean speed, its ground plane 100 m away (ground effect off)",
            s.role == "flank" and abs(of.V - v_side) < 1e-9 and "lap's mean speed" in of.V_source
            and abs(of.ride_band[0] - (am.bridge.OFFSET_M + 0.25)) < 1e-9
            and abs(of.ride_band[1] - (am.bridge.OFFSET_M + 0.70)) < 1e-9,
            f"V {of.V:.3f} m/s ({of.V_source}); ride band {of.ride_band[0]:.2f}-"
            f"{of.ride_band[1]:.2f} m over a plane at {of.deck:.0f} m")
        pol = g.policy
        b_bal = [r_.get("budget") for r_ in pol.plans(s)]
        g.search["effort"] = "quick"
        b_q = [r_.get("budget") for r_ in pol.plans(s)]
        g.search["effort"] = "balanced"
        rep("Search & budget is WingLab's measured plan: 164 / 164 / 53 at balanced, 109 / 109 / 42 "
            "quick (`Garage.search` is a view of the same policy)",
            b_bal == [164, 164, 53] and b_q == [109, 109, 42] and pol.effort == "balanced",
            f"balanced {b_bal}, quick {b_q}")
        V_lap = s.op.V
        re_lap = af.conditions().get("re_own")
        mp.params.param("vauto").set(False)
        mp.params.param("vdes").set(29.0)
        typed = (s.op.V, s.op.V_source, af.conditions().get("re_own"))
        mp.params.param("vauto").set(True)
        rep("Mission ▸ Design point types the slot's speed (29 m/s, the R 100 m limit) and the "
            "sections' Reynolds number follows it; 'from the lap' gives the lap's back",
            typed[:2] == (29.0, "typed") and abs(typed[2] / re_lap - 29.0 / V_lap) < 1e-9
            and abs(s.op.V - V_lap) < 1e-12,
            f"V {V_lap:.2f} -> 29.00 m/s: main Re {re_lap:.4g} -> {typed[2]:.4g}")

        # -- 2 Airfoil: screen, rank, take ----------------------------------
        view("af.screen")
        g._design_shortcut(pygame.K_l)
        job = g.runs.live
        kind = job.info.kind if job is not None else None
        g.runs.run_all()
        out = af.screen["outcome"] or {}
        rep("L on 2 Airfoil ▸ Library screening screens WingLab's section library LIVE (a worker "
            "job, replayed): 14 ranked at the library point",
            kind == "screen" and len(af.ranked) == 14 and out.get("state") == "done"
            and af.screen["re_source"] == "library",
            f"{len(af.ranked)} ranked of {(af.screen['report'] or {}).get('n_eligible')} eligible, "
            f"best {af.ranked[0]['name'] if af.ranked else '-'}; {am.outcome_tag(out)}")
        view("af.rank")
        dp.refresh_rank()
        items = dp.rank_list.items
        heads = [h for _k, h, _f in af.columns()]
        rep("the ranking table: WingLab's top 14 with each criterion's value and points -- and no "
            "cd@cl on the wing (the owner: on a wing it is L/D at the design cl again)",
            len(items) == 14 and "cd@cl" not in heads and all("cd@cl" not in it[1] for it in items)
            and "L/D@cl" in items[0][1] and "(" in items[0][1],
            f"{items[0][0]}: {items[0][1][:84]}" if items else "no rows")
        dp.rank_list.idx = 1
        name1 = af.ranked[1]["name"]
        g._design_shortcut(pygame.K_f)
        fl = w.physics_flags()
        rep("F on the ranking takes the highlighted section: 2 Airfoil is answered, the wing's "
            "flags carry it, the page moves on to 2.8",
            af.decision == "library" and (af.chosen or {}).get("name") == name1
            and fl.get(api.SECTION_KEY) == af.flag_value() and dp.nav.current().startswith("ep."),
            f"{name1} -> flag {fl.get(api.SECTION_KEY)!r}; now at '{dp.nav.current()}'")

        # -- 2.8 Endplate: the family's own plate, then a symmetric one -------
        view("ep.section")
        g._design_shortcut(pygame.K_f)
        fl = w.physics_flags()
        rep("F on 2.8 ▸ Section with nothing screened keeps the family's own plate (NACA 00tt): "
            "an answer, no plate flag, the page moves on to 3 Wing",
            ep.decision == "default" and ep.finished() and api.SECTION_PLATE_KEY not in fl
            and dp.nav.current().startswith("w."), f"{ep.msg}; now at '{dp.nav.current()}'")
        view("ep.screen")
        g._design_shortcut(pygame.K_l)
        g.runs.run_all()
        sym = set(api.symmetric_section_names())
        view("ep.rank")
        dp.refresh_rank()
        rep("2.8 screens SYMMETRIC sections only (the owner's rule) and keeps cd@cl; L/D and |cm| "
            "are dead at cl 0",
            len(ep.ranked) == 14 and all(r_["name"] in sym for r_ in ep.ranked)
            and "cdcr" in [c[0] for c in ep.columns()]
            and sorted(ep.dead()) == ["cm", "ldcr", "ldmax"]
            and not ep.screen_params.param("w.ldcr").enabled,
            f"{len(ep.ranked)} ranked, all in the {len(sym)} symmetric; best "
            f"{ep.ranked[0]['name'] if ep.ranked else '-'}")
        dp.rank_list.idx = 0
        g._design_shortcut(pygame.K_f)
        pin = w.pinned()
        rep("F on 2.8's ranking takes a symmetric plate, and 3 Wing's box FIXES the plate's t/c to "
            "it (not a player option: the owner, 2026-09-25)",
            ep.decision == "library" and "endplate_tc" in pin
            and w.band_source("endplate_tc") == "fixed from 2.8",
            f"{(ep.chosen or {}).get('name')} (t/c {(ep.chosen or {}).get('tc') or 0:.3f}): "
            f"endplate_tc pinned at {pin.get('endplate_tc')}")

        # -- 2 Airfoil ▸ Shape optimisation --------------------------------------
        view("af.opt")
        g._design_shortcut(pygame.K_o)
        job = g.runs.live
        kind = job.info.kind if job is not None else None
        g.runs.run_all()
        out = dict(af.opt["outcome"] or {})
        gr = af.graph()
        rep("O on 2 Airfoil ▸ Shape optimisation runs WingLab's section search on the worker "
            "(replayed): DONE, one dot per evaluation on the graph, the Sobol block marked",
            kind == "section" and out.get("state") == "done" and gr["N"] == len(gr["records"])
            == out.get("k") and gr["n_init"] == 4 and af.has_optimised(),
            f"{am.outcome_tag(out)}; graph N {gr['N']}, {len(gr['records'])} dots, n_init "
            f"{gr['n_init']}")
        prev = af.spent()
        g._design_shortcut(pygame.K_k)
        g.runs.run_all()
        out = dict(af.opt["outcome"] or {})
        rep("K keeps going: WingLab's resume -- the evaluations already paid for are inherited, "
            "nothing is re-flown, the counter goes on",
            out.get("state") == "done" and out.get("n_prior") == prev == af.opt["prior"]
            and len(af.opt["records"]) == out.get("k") > prev,
            f"{prev} -> {out.get('k')} evaluations ({out.get('n_prior')} inherited)")
        rp.pause_at = 5
        g._design_shortcut(pygame.K_o)
        job = g.runs.live
        if job is not None:
            job.settle(10.0)
        key(pygame.K_ESCAPE)
        st1, page1 = (job.state if job is not None else None), g.page
        g.runs.run_all()
        rp.pause_at = None
        out = dict(af.opt["outcome"] or {})
        key(pygame.K_ESCAPE)
        back = g.page
        rep("ESC during a run is Stop: STOPPED · 5/N with the best so far kept and usable; once it "
            "has ended ESC steps back to the mission",
            st1 == "stopping" and page1 == "section" and out.get("state") == "stopped"
            and out.get("k") == 5 and af.has_optimised() and back == "mission",
            f"{st1} on {page1}; {am.outcome_tag(out)}; then ESC -> {back}")
        view("af.opt")
        g._design_shortcut(pygame.K_f)
        fv = af.flag_value() or {}
        rep("F on Shape optimisation takes the optimised section: the wing flies its CST weights "
            "at the point it was designed at",
            af.decision == "optimised" and isinstance(fv, dict) and bool(fv.get("w_upper"))
            and bool(fv.get("re")), f"{(af.chosen or {}).get('origin')}")

        # -- 3 Wing -------------------------------------------------------------
        view("w.type")
        objs = w.objectives()
        rep("3 Wing ▸ Wing type offers WingLab's car objectives for the slot -- the flank's "
            "downforce is side force, and no lap time (cartrack cannot value a lateral device)",
            list(objs) == ["efficiency", "downforce", "drag", "downforce_plus_drag"]
            and "side force" in objs["downforce"]
            and w.type_params.param("obj").choices == list(objs)
            and w.choices["objective"] == "downforce", f"{list(objs)}; '{objs['downforce']}'; "
            f"opens on {w.choices['objective']} (the side job's own)")
        #  the captured flank wing fixture was flown at AeroBO's default,
        #  efficiency: the player picks it here, as they may on any job
        w.type_params.param("obj").set("efficiency")
        view("ep.screen")
        w.type_params.param("plates").set("fences")
        dp.settle()
        locked = (dp.plate_locked(), dp._state("ep.screen"), dp.nav.current(),
                  [r_.get("stage") for r_ in pol.plans(s)])
        w.type_params.param("plates").set("designed")
        rep("endplates: fences LOCKS 2.8 (nothing there to give a section to), moves the cursor "
            "off it and drops it from the Search table; designed opens it again",
            locked[0] and locked[1] == "locked" and not locked[2].startswith("ep.")
            and "2.8 Endplate" not in locked[3] and not dp.plate_locked()
            and dp._state("ep.screen") != "locked", f"{locked[1]}, cursor at '{locked[2]}', "
                                                   f"table {locked[3]}")
        view("w.box")
        form = dp.rows()
        labels = list(w.family_box())
        keys_ = [p_.key for p_ in form.params]
        #  the owner, 2026-09-25: "endplate t/c shouldn't be given as an
        #  option" -- its row is shown read-only, with no controls at all
        rep("3 Wing ▸ Design box: min / max / constrain / fix for every row of WingLab's built "
            "problem (the 14 of 'car rear wing + endplates + free chord law') but endplate_tc, "
            "which has none",
            form is w.box_params() and len(labels) == 14 and "boxreset" in keys_
            and all(f"bx.{lab}.{t_}" in keys_ for lab in labels if lab != "endplate_tc"
                    for t_ in ("min", "max", "con", "fix"))
            and "endplate_tc" in labels
            and not any(k_.startswith("bx.endplate_tc.") for k_ in keys_),
            f"{len(labels)} rows: {', '.join(labels[:5])}, ...")
        band = list(w.default_band("taper"))
        form.param("bx.taper.max").set(0.8)
        bo1 = (w.bounds_overrides() or {}).get("taper")
        form.param("bx.taper.con").set(False)
        bo2, src2 = (w.bounds_overrides() or {}).get("taper"), w.band_source("taper")
        form.param("bx.twist_root_deg.fix").set(True)
        pin = dict(w.pinned())
        form.param("bx.twist_root_deg.fix").set(False)
        form.param("boxreset").activate()
        rep("the box's switches are WingLab's: a narrowed band reaches bounds_overrides, released "
            "drops it, fixed pins the row; reset gives the family's box back",
            bo1 == [band[0], 0.8] and bo2 is None and src2 == "released"
            and "twist_root_deg" in pin and "twist_root_deg" not in w.pinned()
            and not w.box and not w.released and not w.fixed,
            f"taper {band} -> {bo1} -> released; twist_root_deg pinned at "
            f"{pin.get('twist_root_deg')}")
        view("w.solver")
        cfg = w.cfg()
        f_ = cfg.flags or {}
        plan = w.plan()
        rep("3 Wing ▸ Solver: V3's configuration for the slot family -- bo_slsqp at WingLab's "
            "measured budget for the rows searched, carsim's V, both chosen sections in the flags",
            cfg.problem_name == w.family_name and cfg.optimiser == "bo_slsqp"
            and int(cfg.budget) == int(plan.budget) and abs(float(f_["V"]) - s.op.V) < 1e-9
            and f_.get(api.SECTION_KEY) == af.flag_value()
            and f_.get(api.SECTION_PLATE_KEY) == ep.flag_value()
            and "endplate_tc" in (cfg.pinned or {}),
            f"{cfg.problem_name}: {cfg.optimiser}, {cfg.budget} evaluations for {plan.dim} rows "
            f"(the plate's t/c fixed from 2.8), V {float(f_['V']):.2f} m/s")
        g._design_shortcut(pygame.K_o)
        job = g.runs.live
        kind, at = (job.info.kind if job is not None else None), dp.nav.current()
        g.runs.run_all()                    # the run, then the law, then the design report
        out = dict(w.outcome or {})
        rep("O on 3 Wing ▸ Solver launches the wing run LIVE and shows Convergence; it ends DONE, "
            "then WingLab's law and design report follow by themselves and 4 Results opens",
            kind == "wing" and at == "w.conv" and out.get("state") == "done"
            and w.record is not None and w.law is not None and dp.results.report is not None
            and dp.nav.open("r.summary"),
            f"{am.outcome_tag(out)}; best {w.record.get('best_score') if w.record else '-'} "
            f"{(w.record or {}).get('score_units') or ''}")
        rp.pause_at = 8
        g._design_shortcut(pygame.K_o)
        job = g.runs.live
        if job is not None:
            job.settle(10.0)
        chip = job.chip() if job is not None else None
        w.conv_params.param("stop").activate()
        g.runs.run_all()
        rp.pause_at = None
        o_stop = dict(w.outcome or {})
        w.conv_params.param("more").set(6)
        w.conv_params.param("go").activate()
        g.runs.run_all()
        rec = w.record or {}
        rep("Stop on Convergence after 8 evaluations keeps the best (STOPPED · 8/N); Keep going +6 "
            "resumes it -- 8 inherited, nothing re-flown",
            chip is not None and " 8/" in chip[0] and o_stop.get("state") == "stopped"
            and o_stop.get("k") == 8 and int(rec.get("resumed") or 0) == 8
            and int(rec.get("n_evals") or 0) == 14 and len(w.records) == 14,
            f"{chip[0] if chip else '-'} -> {am.outcome_tag(o_stop)} -> "
            f"{am.outcome_tag(w.outcome)}; resumed {rec.get('resumed')}")

        # -- 4 Results ---------------------------------------------------------
        sm = dp.results.summary()
        rep("4 Results ▸ Summary: WingLab's numbers (score, CZ, CD, side force, drag) and the "
            "car's -- the game's force at V is WingLab's",
            bool(sm) and sm["force_word"] == "side force" and sm.get("CZ") is not None
            and sm.get("aerobo_force_N") and abs(sm["game_force_N"] - sm["aerobo_force_N"])
            <= 1e-9 * abs(sm["aerobo_force_N"]),
            f"best {sm.get('best_score')} {sm.get('score_units')}; F {sm.get('force_N')} N, "
            f"game {sm.get('game_force_N')} N; {len(sm.get('margins') or [])} margins")
        cl = dp.results.car_lap()
        rep("...and carsim's own check on the side wing's circuit -- the lap with and without "
            "the wing -- labelled as the second model it is",
            cl.get("kind") == "lap" and cl.get("with") is not None
            and cl.get("without") is not None and "second model" in cl.get("model", "")
            and g.mission.side_track in cl.get("title", ""),
            f"{cl.get('with') or 0:.3f} s vs {cl.get('without') or 0:.3f} s without "
            f"({cl.get('delta') or 0:+.3f} s)")

        # -- onto the car ----------------------------------------------------------
        g.prompt_rename()
        was_open = g.prompt.open
        g.prompt.value = ""
        for ch in "flank-aerobo":
            g._handle(pygame.event.Event(pygame.KEYDOWN, key=ord(ch) if ch != "-" else pygame.K_MINUS,
                                         mod=0, unicode=ch))
        g._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))
        rep("N names the fitted wing before it goes on the car",
            was_open and w.spec is not None and w.spec.name == "flank-aerobo" and w.dirty,
            w.spec.name if w.spec is not None else "-")
        n_af = set(lib.airfoils)
        g._design_shortcut(pygame.K_s)
        name = w.spec.name if w.spec is not None else ""
        spec_l = lib.wings.get(name)
        new_secs = sorted(set(lib.airfoils) - n_af)
        origins = [lib.airfoils[n_].origin for n_ in new_secs]
        rep("S puts it on the car: the wing (engine WingLab, its problem and winning vector kept) "
            "and the sections it flies (as coordinates, with their WingLab origin) go into the "
            "library; both flanks take it at WingLab's incidence",
            spec_l is not None and spec_l.engine == "aerobo" and bool(spec_l.design.get("x"))
            and g.build.left.wing == name == g.build.right.wing and not w.dirty
            and abs(g.build.left.inc_deg - float(w.slot_updates["inc_deg"])) < 1e-9
            and len(new_secs) == 2 and all(o_.startswith("WingLab") for o_ in origins)
            and spec_l.airfoil in lib.airfoils and spec_l.plate_airfoil in lib.airfoils,
            f"'{name}' at inc {g.build.left.inc_deg:+.2f} deg; sections {new_secs}")
        kw = g.build.cfg_kwargs(lib)
        dev = kw.get("dev_left")
        law = spec_l.aero if spec_l is not None else {}
        V_ = float(law.get("V_ref") or 0.0)
        F_car = 0.5 * RHO * V_ ** 2 * dev.S * dev.cl(dev.inc) if dev is not None else 0.0
        F_ab = float((law.get("aerobo") or {}).get("F_N") or 0.0)
        rep("the car flies WingLab's law: the flank's DevAero at the slot's incidence makes WingLab's "
            "side force at V exactly",
            dev is not None and dev.name == name and kw.get("dev_right") is not None
            and F_ab > 0.0 and abs(F_car - F_ab) <= 1e-9 * F_ab,
            f"F car {F_car:.6f} N vs WingLab {F_ab:.6f} N at {V_:.2f} m/s")
        g.page = "car"
        g._select("left")
        key(pygame.K_RIGHT)
        stale = am.law_stale(lib.wings[name], g.build.left)
        done = g.rederive_stale()
        law2 = lib.wings[name].aero
        dev2 = g.build.cfg_kwargs(lib)["dev_left"]
        F2 = 0.5 * RHO * float(law2["V_ref"]) ** 2 * dev2.S * dev2.cl(dev2.inc)
        rep("moving the slot on the car page makes the law stale; before the build leaves the "
            "garage it is re-derived through WingLab at the new station",
            stale and done == ["left"] and not am.law_stale(lib.wings[name], g.build.left)
            and abs(float(law2["derived_at"]["x"]) - g.build.left.x) < 1e-12
            and abs(F2 - float(law2["aerobo"]["F_N"])) <= 1e-9 * abs(F2),
            f"x -> {g.build.left.x:+.2f} m: re-derived {done}, F {F2:.3f} N")
        rep("the right flank is the left one's design (mirrored): one session, one wing",
            dp.session_for("right") is dp.session_for("left") and g.build.right.wing == name, name)

        # -- task 41: the car's span limit holds the designed wing ----------------
        #  (a) the flank's Design box is the Corsa's: its span row opens at the
        #  limit at this slot height (the lower tip at the 0.15 m ground
        #  clearance); a max typed past it is held at that ceiling, at 3x it
        #  in Unlimited -- the search never gets more
        h_l = g.build.left.h
        s.refresh_op()
        lim_l = 2.0 * (h_l - 0.15)
        row_l = list((s.op.size_rows or {}).get("b_m") or [0.0, 0.0])
        w._box_set("b_m", 1)(9.0)
        b_real = w.bounds_overrides()["b_m"][1]
        g.settings = _NS(wing_limits="unlimited")
        s.refresh_op()
        w._box_set("b_m", 1)(9.0)
        b_unl = w.bounds_overrides()["b_m"][1]
        g.settings = None
        s.refresh_op()
        w.box.pop("b_m", None)
        rep("the flank's Design box is the car's: its span row opens at the Corsa's limit at this "
            "height, a typed 9 m is held at it (Real) or at 3x it (Unlimited)",
            abs(row_l[1] - lim_l) < 1e-6 and abs(b_real - lim_l) < 1e-9
            and abs(b_unl - 3.0 * lim_l) < 1e-9 and s.op.limit == lim_l,
            f"h {h_l:.2f}: row {row_l}, typed 9 -> {b_real:.3f} m Real, {b_unl:.3f} m Unlimited")
        #  (b) S in Real mode fits no wing past the limit in ANY slot that
        #  would carry it: mirror off, the same wing on a right flank at h 0.40
        #  (limit 0.50 m) -- refused, saying which slot, its limit and where
        #  Unlimited is; nothing written. Unlimited saves it, filed apart
        g.build.mirror = False
        g.build.right.h = H_W_MIN
        span_w = float(w.spec.span)
        before = lib.wings[name].to_json()
        got_r = w.commit()
        msg_r = w.msg
        same_r = lib.wings[name].to_json() == before
        g.settings = _NS(wing_limits="unlimited")
        got_u = w.commit()
        over_u = bodies.over_limits(g.build, lib, "corsa")
        g.settings = None
        g.build.mirror = True
        g.build.sync_mirror("left")
        g.build.clamp(lib, g.car)
        rep("Real: S refuses a designed wing past the limit in any slot that carries it (the "
            "right flank at h 0.40: 0.50 m), says why and where Unlimited is; Unlimited saves it",
            span_w > 0.50 and got_r == "" and "right slot's 0.50 m" in msg_r
            and "Unlimited" in msg_r and same_r and got_u == name
            and [o_["slot"] for o_ in over_u] == ["right"],
            f"span {span_w:.3f} m: {msg_r[:90]} ...; Unlimited: saved, over {bodies.limits_text(over_u)}")

        # -- the policy, XFOIL, the lock -------------------------------------------
        g.open_section("left")
        view("w.solver")
        g.search["mode"] = "own"
        w.solver_params.param("budget").set(30)
        b_own = w.search()["budget"]
        g.search["mode"] = "recommended"
        rep("Search & budget ▸ own: the player's budget reaches the run; recommended gives "
            "WingLab's back", b_own == 30 and w.search()["budget"] == w.plan().budget,
            f"own {b_own}, recommended {w.search()['budget']}")
        am.use_engine()
        try:
            view("af.opt")
            refused = not af.start_optimise()
            why = af.msg
        finally:
            rp = am.use_fixtures()
        rep("without XFOIL and without a replay, shape optimisation is refused with WingLab's "
            "reason, and the screen falls back to the library point",
            refused and "XFOIL" in why and af.effective_re_source() == "library", why[:80])
        rp.pause_at = 3
        view("af.opt")
        g._design_shortcut(pygame.K_o)
        job = g.runs.live
        if job is not None:
            job.settle(10.0)
        before = (json.dumps(g.build.to_json(), sort_keys=True), g.mission.track, dp.opened_for,
                  len(lib.wings))
        stated = g.state_mission()
        g._menu_action("defaults")
        g._menu_action("airfoils")
        second = w.start_run()
        after = (json.dumps(g.build.to_json(), sort_keys=True), g.mission.track, dp.opened_for,
                 len(lib.wings))
        still = g.runs.live is job and job is not None
        g.runs.stop()
        g.runs.run_all()
        rep("while a run is live, stating the mission, the menu's defaults and the airfoil page "
            "are refused, a second run is refused (one engine thread), and the run goes on",
            not stated and before == after and still and not second and g.page == "section",
            f"state {stated}, second run {second}, run on {still}")
        rep_before = af.opt["report"]
        g._design_shortcut(pygame.K_o)
        job = g.runs.live
        if job is not None:
            job.settle(10.0)
        dp.open("left")
        g.runs.run_all()
        rp.pause_at = None
        rep("re-opening the design page abandons a live run: nothing it found is applied, and "
            "nothing is left running",
            job is not None and job.abandoned and g.runs.live is None and not g.runs.busy
            and af.opt["report"] is rep_before, f"job {job.state if job else '-'}")
        #  a live run's frames: the replay parks at 10 evaluations, and the
        #  garage frame draws Convergence with its live graph around it
        rp.pause_at = 10
        view("w.solver")
        g._design_shortcut(pygame.K_o)
        job = g.runs.live
        if job is not None:
            job.settle(10.0)
        ts = []
        for _ in range(6):
            t0 = time.perf_counter()
            g.frame(1.0 / 60.0)
            ts.append((time.perf_counter() - t0) * 1e3)
        live_ok = g.runs.live is job and job is not None and dp.nav.current() == "w.conv"
        g.runs.stop()
        g.runs.run_all()
        rp.pause_at = None
        ok12, why12 = _budget(max(ts), 90.0)
        rep("a live wing run's frames (the garage frame drawing Convergence) stay within budget",
            ok12 and live_ok and "w.conv" not in g.shell.view_errors,
            f"{why12}; worst {max(ts):.1f} ms over {len(ts)} frames at 10 evaluations")
        binds = []
        for k_, want in (("af.screen", af.screen_params), ("af.section", af.params),
                         ("af.opt", af.opt_params), ("ep.screen", ep.screen_params),
                         ("w.type", w.type_params), ("w.box", w.box_params()),
                         ("w.solver", w.solver_params), ("w.conv", w.conv_params),
                         ("r.summary", dp.results.params)):
            view(k_)
            binds.append(dp.rows() is want)
        view("af.rank")
        binds.append(dp.rows() is None)
        rep("each view binds its model's form (DesignPage.rows); the ranking's table is rank_list",
            all(binds), f"{sum(binds)} of {len(binds)}")

        # -- the top slot ------------------------------------------------------------
        g.open_mission("top")
        g.state_mission()
        st_ = dp.session
        wt = st_.wing
        wt.type_params.param("plates").set("fences")
        wt.type_params.param("obj").set("laptime")
        fam = wt.family
        wt.type_params.param("plates").set("designed")
        rep("the top slot has its own session on WingLab's top family: ground effect on over the "
            "boot, lap time round carsim's circuit offered with plain fences only",
            st_.role == "top" and st_ is not s and abs(st_.op.deck - deck_z(g.build.top.x)) < 1e-12
            and abs(st_.op.ride_band[0] - (st_.op.deck + 0.14)) < 1e-12
            and fam.lap is not None and fam.lap[0] == g.mission.track and not fam.plates
            and "laptime" not in wt.objectives() and wt.choices["objective"] == "efficiency",
            f"deck {st_.op.deck:.3f} m, ride {st_.op.ride_band[0]:.2f}-{st_.op.ride_band[1]:.2f} m; "
            f"lap {fam.lap}")
        g.open_mission("left")
        g.state_mission()
        ranked0 = [r_["name"] for r_ in af.ranked]
        calls = []
        real_open = dp.open
        dp.open = lambda k_: (calls.append(k_), real_open(k_))[1]
        try:
            keeps = g.mission_restate_keeps()
            restated = g.state_mission()
        finally:
            del dp.open
        rep("re-stating an unchanged mission keeps the slot's session -- its screens, sections and "
            "runs (WingLab's accept is idempotent)",
            keeps and restated and not calls and dp.session is s and w.record is not None
            and [r_["name"] for r_ in af.ranked] == ranked0 and ranked0,
            f"{len(ranked0)} still ranked, open called {len(calls)} times")

        # --- the airfoil page
        key(pygame.K_a)
        rep("A opens the airfoil page ranked", g.page == "airfoil" and len(g.af_page.list.items) >= 30,
            f"{len(g.af_page.list.items)} sections, top {g.af_page.list.items[0][0]}")
        t0 = time.perf_counter()
        g.frame(1.0 / 60.0)
        ms_ = (time.perf_counter() - t0) * 1e3
        rep("airfoil page draws", *_budget(ms_, 80.0))
        if screenshot_dir:
            pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_airfoil.png"))
        names_ = [it[0] for it in g.af_page.list.items]
        known = [i for i, n_ in enumerate(names_) if api.library_section_available(n_)]
        g.af_page.list.idx = known[0] if known else 0
        pick = names_[g.af_page.list.idx]
        key(pygame.K_RETURN)
        rep("ENTER on a section WingLab's library also holds hands it to 2 Airfoil by name (at the "
            "library point) and returns to the DESIGN page",
            g.page == "section" and af.decision == "library" and (af.chosen or {}).get("name") == pick
            and (af.chosen or {}).get("library_point") and dp.nav.current() == "af.section", pick)
        #  ESC steps BACK through the chain rather than dropping to the car:
        #  design -> mission -> car, and a dirty wing is saved on the way out
        w.fit("flank-aerobo-b")
        key(pygame.K_ESCAPE)
        rep("ESC leaves the design page, saves the dirty wing into the library and the slot, and "
            "steps back to the mission",
            g.page == "mission" and not w.dirty and "flank-aerobo-b" in lib.wings
            and g.build.left.wing == "flank-aerobo-b", f"{g.page}, left {g.build.left.wing}")
        had_ = (af.decision, w.record is not None)
        g.mission.track, g.mission.stated = "open", False           # the TOP wing's job
        restated2 = g.state_mission()
        kept_ = (af.decision, w.record is not None)
        g.mission.track, g.mission.stated = "arena", False
        g.state_mission()
        rep("the top wing's changed circuit KEEPS a side wing's session when the mission is "
            "stated again: the side wing flies its own circuit",
            restated2 and kept_ == had_ and had_ == ("library", True) and dp.session is s,
            f"before {had_}, after {kept_}")
        key(pygame.K_ESCAPE)
        key(pygame.K_ESCAPE)
        rep("ESC twice steps back to the car", g.page == "car", g.page)
    finally:
        am.use_engine()
        if env_nx is None:
            os.environ.pop("CARSIM_NO_XFOIL", None)
        else:
            os.environ["CARSIM_NO_XFOIL"] = env_nx
    # --- the library page (task 46: SHIFT+L; L is SAVED WINGS)
    key(pygame.K_l, pygame.KMOD_SHIFT)
    rep("SHIFT+L opens the library (the text list)", g.page == "library"
        and len(g.lib_page.wings.items) >= 5, f"{len(g.lib_page.wings.items)} wings")
    g.frame(1.0 / 60.0)
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_library.png"))
    g.prompt_build()
    g.prompt.value = ""                       # the prompt opens on the current name
    for ch in "test-car":
        g._handle(pygame.event.Event(pygame.KEYDOWN, key=ord(ch) if ch != "-" else pygame.K_MINUS, mod=0, unicode=ch))
    g._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))
    rep("save build via the prompt", "test-car" in lib.builds and g.lib_page.builds.items, str(list(lib.builds)))
    g.lib_page.focus = "builds"
    g.build.reset()
    g._library_select()
    rep("loading the build restores the car (the WingLab wing on both flanks)",
        g.build.left.wing == "flank-aerobo-b" == g.build.right.wing and g.build.top.wing == "rear-s1223",
        g.build.summary(lib)[:60])
    ok = _check_builds(g, lib, key, rep) and ok
    ok = _check_confirms(g, lib, key, rep) and ok
    ok = _check_hints(g, rep) and ok
    ok = _check_build_header(g, lib, key, rep) and ok
    ok = _check_car_panel(g, lib, rep) and ok
    ok = _check_saved(g, lib, rep, screenshot_dir) and ok        # task 46
    # --- the pad guard, as before
    class _FakePad:
        name, layout = "DualSense Wireless Controller", "ps"

        def __init__(self):
            self.held = set()

        def pressed(self, n):
            return n in self.held

        def stick(self, which="left"):
            return (0.0, 0.0)

    fp = _FakePad()
    fp.held.add("cross")
    g2 = Garage((1280, 800), CarBuild(), pad=fp, headless=True, lib=lib)
    acts = [g2.frame(1.0 / 60.0) for _ in range(30)]       # 0.5 s, cross held
    rep("held CROSS at garage start does not drive", not any(acts), str(set(acts)))
    fp.held.discard("cross")
    g2.frame(1.0 / 60.0)
    fp.held.add("cross")
    rep("a fresh CROSS press drives", g2.frame(1.0 / 60.0) == "drive", "")
    fp.held.clear()
    g2.frame(1.0 / 60.0)
    fp.held.add("l3")
    g2.frame(1.0 / 60.0)
    rep("L3 opens the MISSION on the pad (the chain's entry point)",
        g2.page == "mission", g2.page)
    fp.held.clear()
    g2.frame(1.0 / 60.0)
    fp.held.add("circle")
    g2.frame(1.0 / 60.0)
    rep("CIRCLE backs out of a page", g2.page == "car", g2.page)
    g3 = Garage((1280, 800), CarBuild(), headless=True, lib=lib)
    early = g3._handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    rep("ENTER inside the first 0.35 s is swallowed by frame()",
        early == "drive" and g3.frame(1.0 / 60.0) is None, "")
    a = key(pygame.K_ESCAPE)
    rep("ESC on the car page opens the menu", a is None and g.menu.open, f"{a} open={g.menu.open}")
    g.frame(1.0 / 60.0)
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_menu.png"))
    #  task 45: the menu's exits -- 'Main menu' (back to the title) right above
    #  'Quit to desktop', which asks twice; the design row says what it does
    acts = [a_ for _, a_ in g.menu.items]
    lbls = dict((a_, lbl) for lbl, a_ in g.menu.items)
    rep("the menu's exits: 'Drive this car', 'Main menu' (title), 'Quit to desktop' last; "
        "the design row 'Design a wing for the <slot> slot'",
        acts[-3:] == ["drive", "title", "quit"] and lbls["title"] == "Main menu"
        and lbls["quit"] == "Quit to desktop"
        and lbls["design"] == f"Design a wing for the {g.sel} slot", str(acts[-3:]))

    def tap_(k):                      # a press and its release: a fresh second press
        r_ = key(k)
        g._handle(pygame.event.Event(pygame.KEYUP, key=k, mod=0))
        return r_

    for _ in range(acts.index("quit")):
        key(pygame.K_DOWN)
    a1 = tap_(pygame.K_RETURN)
    q1 = dict((a_, lbl) for lbl, a_ in g.menu.items)["quit"]
    armed = a1 is None and g.menu.open and g.menu.action() == "quit"
    a2 = tap_(pygame.K_RETURN)
    rep("menu 'Quit to desktop': the first ENTER asks (the row reads 'Quit to desktop: ENTER "
        "again', cursor kept), the second returns quit",
        armed and q1 == "Quit to desktop: ENTER again" and a2 == "quit" and not g.menu.open,
        f"{a1} {q1!r} then {a2}")
    g._menu_open(at="quit")
    tap_(pygame.K_RETURN)
    tap_(pygame.K_UP)                 # off the row: disarmed, the label back
    moved = (g._armed is None and g.menu.action() == "title"
             and dict((a_, lbl) for lbl, a_ in g.menu.items)["quit"] == "Quit to desktop")
    a3 = tap_(pygame.K_RETURN)
    rep("an armed Quit is dropped by moving off the row; 'Main menu' returns title",
        moved and a3 == "title" and not g.menu.open, str(a3))
    #  ... and through the real loop: run() goes on past the first Quit and
    #  ends on the second (posted 4 frames later); a runaway posts a QUIT
    n_, fr = [0], g.frame

    def counted(dt):
        n_[0] += 1
        if n_[0] == 5:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_RETURN, mod=0))
        elif n_[0] == 120:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return fr(dt)

    g.frame = counted
    pygame.event.clear()
    g._menu_open(at="quit")
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_RETURN, mod=0))
    a4 = g.run()
    del g.frame
    rep("run(): the first 'Quit to desktop' does not end it, the second does ('quit')",
        a4 == "quit" and n_[0] == 5, f"{a4} after {n_[0]} frames")
    #  Change car (2026-09-27): the drive's garage has the row, right under
    #  Resume, saying the car; it opens the CHANGE CAR list (every car, this
    #  one marked and under the cursor); Back goes back to the menu on the
    #  row; this car again only says so; another returns 'car' with the pick
    #  in car_wanted. A bare garage (no Settings) has no row.
    import cars as _cars
    g5 = Garage((1280, 800), CarBuild(), headless=True, lib=lib, car="rally",
                settings=_FakeSettings())
    g5._menu_open()
    rows5 = [(lbl, a_) for lbl, a_ in g5.menu.items]
    row_ok = rows5[1] == ("Change car  (now: Halcón)", "cars")
    r_open = g5._menu_action(g5.menu.handle("nav_down") or g5.menu.handle("select"))
    list5 = [a_ for _, a_ in g5.menu.items]
    list_ok = (r_open is None and g5.menu.open and g5.menu.title == "CHANGE CAR"
               and list5 == ["car:" + k for k in _cars.CAR_ORDER] + ["car_back"]
               and g5.menu.action() == "car:rally"
               and g5.menu.items[list5.index("car:rally")][0].endswith("(this car)")
               and "car:mx5" not in list5 and "car:bus" not in list5)
    g5.menu.idx = len(g5.menu.items) - 1                        # Back
    r_back = g5._menu_action(g5.menu.handle("select"))
    back_ok = (r_back is None and g5.menu.open and g5.menu.title == "GARAGE"
               and g5.menu.action() == "cars")
    g5._menu_action(g5.menu.handle("select"))
    r_esc = g5._menu_action(g5.menu.handle("back"))       # ESC / CIRCLE in the list
    back_ok = (back_ok and r_esc is None and g5.menu.open and g5.menu.title == "GARAGE"
               and g5.menu.action() == "cars")
    r_res = g5._menu_action(g5.menu.handle("back"))            # ESC on the menu: resume
    back_ok = back_ok and r_res is None and not g5.menu.open
    g5._menu_open(at="cars")
    g5._menu_action(g5.menu.handle("select"))
    r_same = g5._menu_action(g5.menu.handle("select"))          # the rally car again
    same_ok = r_same is None and g5.car_wanted is None and "already" in g5.hint
    g5._menu_open(at="cars")
    g5._menu_action(g5.menu.handle("select"))
    g5.menu.handle("nav_down")                                  # the rally car -> the 540i
    r_pick = g5._menu_action(g5.menu.handle("select"))
    pick_ok = r_pick == "car" and g5.car_wanted == "540i" and not g5.menu.open
    g6 = Garage((1280, 800), CarBuild(), headless=True, lib=lib)
    g6._menu_open()
    bare_ok = "cars" not in [a_ for _, a_ in g6.menu.items]
    g7 = Garage((1280, 800), CarBuild(), headless=True, lib=lib, car="rally",
                settings=_FakeSettings())
    g7.car_fixed = True                                         # a challenge's car
    g7._menu_open()
    bare_ok = bare_ok and "cars" not in [a_ for _, a_ in g7.menu.items]
    rep("Change car: the menu's row under Resume says the car; its list has every car, "
        "this one marked and under the cursor; Back and ESC are the menu again; this car only "
        "says so; another returns 'car' (car_wanted); no row in a bare garage or on a "
        "challenge's car",
        row_ok and list_ok and back_ok and same_ok and pick_ok and bare_ok,
        f"row {row_ok} list {list_ok} back {back_ok} same {same_ok} "
        f"pick {r_pick}/{g5.car_wanted} bare {bare_ok}")
    #  task 45: the menu with a pad's help fits the window (it ran 15 px off)
    ok = _check_menu_fits(lib, _FakePad(), rep) and ok
    # --- the AeroBO shell at 1600 x 1000 sits on AeroBO's measured rectangles
    from .cae import chrome as C
    from .cae import theme as T
    g4 = Garage((1600, 1000), CarBuild(), headless=True, lib=lib)
    g4.aerobo_dir = None
    g4.open_mission("left")
    g4.state_mission()
    g4._draw_page()
    L4, want = g4.shell._L, C.shell_rects(1600, 1000)
    bad = [k_ for k_, r_ in want.items() if getattr(L4, k_) != r_]
    ys = [r_.y for _rid, r_, kind_ in g4.shell._tree_hits if kind_ == "row" and r_.h >= T.TREE_ROW_H]
    rep("the 1600x1000 shell sits on WingLab's measured rectangles; tree rows 21 px apart from y 86",
        not bad and ys and ys == [86 + 21 * i for i in range(len(ys))],
        "; ".join(bad) or f"{len(want)} rects, {len(ys)} tree rows from y {ys[0] if ys else '-'}")
    shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"  {n_rows[1]}/{n_rows[0]} " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
