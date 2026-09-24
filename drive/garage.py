"""3D garage: build the car's three wings -- a panel on each flank and a
wing on top -- design them from the section up, keep them in a library,
and drive exactly that car.

What changed from the one-panel editor
--------------------------------------
The published study has ONE flank panel (S 0.35 m^2, CL0 0.70 / 1.25, L/D
3.2) placed by four numbers. That car is still here, bit-for-bit, as the
built-in wings 'fin' and 'plate'. Around it:

* a BUILD has three slots -- left flank, right flank, top -- each holding
  a wing from the library at a station, a height and an incidence. Left
  and right mirror each other unless you unlock them.
* a WING is a WingSpec (drive/aero/wing.py): section, span, chord, taper,
  twist, end plates. It is designed in two gated pages, AeroBO's order:
  the MISSION page (a lap of one of carsim's circuits, stated before
  anything is judged) and the DESIGN navigator -- four groups of four
  steps, AIRFOIL / ENDPLATE / WING / RESULTS, each step open only once the
  one before it is finished. The lattice runs live (the AeroBO car-wing
  physics, numpy port) and every search is a GP-BO. The AIRFOIL page ranks
  the section library and runs XFOIL in the background; the LIBRARY page
  saves and loads wings and whole builds, so a wing drawn for one car goes
  on the next. Every page takes the mouse as well as the keyboard.
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
from dataclasses import dataclass, asdict, field

import numpy as np
import pygame

from corsa_c import CorsaC, RHO, G
import crossover
from .menu import Menu, StickNav
from . import garage_ui as ui
from .aero.library import Library
from .aero.wing import (WingSpec, BOUNDS, V_REF, RIDE_H0, design_point, spanwise,
                        re_bank_snap, design_bounds, design_x0,
                        design_labels, apply_design, span_fit, format_design,
                        design_vars, design_table,
                        MOUNTS, MOUNTS_BUILDABLE, MOUNT_PLATE_H, wing_mass, TOP_PYLON_L)
from .aero import optimize as opt
from .aero import airfoil as af
from .aero import mission as ms
from .aero import section as sec
from .aero import blend as bl
from .aero import screen as scr
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
SLOT_LABEL = {"left": "LEFT FLANK", "right": "RIGHT FLANK", "top": "TOP WING"}
SLOT_ROLE = {"left": "flank", "right": "flank", "top": "top"}
#  The two DESIGN pages are walked in order -- mission, then the navigator --
#  and the order is enforced, not suggested (`Garage.open_section`). 'airfoil'
#  and 'library' are not steps: they browse what already exists.
PAGES = ("car", "mission", "section", "airfoil", "library")
#  the two DESIGN pages, in order. 'section' is the navigator over everything
#  downstream of the mission -- airfoil, end plate, wing and results -- so the
#  old separate wing page is a GROUP of it now, not a page.
DESIGN_STEPS = ("mission", "section")

# the pause menu's help columns (ESC / OPTIONS)
GARAGE_HELP_KB = [
    ("mouse drag / wheel", "orbit / zoom"),
    ("1 / 2 / 3, TAB", "select slot: left / right / top"),
    ("LEFT / RIGHT", "station x  (SHIFT: 1 cm)"),
    ("UP / DOWN", "height h  (SHIFT: 1 cm)"),
    ("[ / ]", "incidence -1 / +1 deg"),
    ("W / SHIFT+W", "next / previous library wing in the slot"),
    ("M", "mirror left <-> right"),
    ("T", "top wing: fixed / active (brake+steer)"),
    ("SPACE", "deploy preview (0.45 s actuator)"),
    ("D", "design a wing: mission -> section -> wing"),
    ("A / L", "airfoil library / wing + build library"),
    ("R / C", "car defaults / reset camera"),
    ("ENTER", "drive this car"),
    ("H", "wing tutorial's box: hide / show"),
    ("mouse", "in this menu: point, click a row"),
    ("ESC", "this menu"),
]
GARAGE_HELP_PAD = [
    ("left stick", "move the wing (x, h)"),
    ("right stick", "orbit"),
    ("d-pad", "step x / h"),
    ("L1 / R1", "incidence -1 / +1 deg"),
    ("TRIANGLE", "next slot"),
    ("SQUARE", "next library wing in the slot"),
    ("CIRCLE", "deploy preview"),
    ("L3", "design a wing for this slot (mission first)"),
    ("R3", "reset camera"),
    ("CROSS", "drive this car"),
    ("OPTIONS", "this menu"),
]
GARAGE_MENU_KEYS = {
    pygame.K_UP: "nav_up", pygame.K_DOWN: "nav_down",
    pygame.K_RETURN: "select", pygame.K_KP_ENTER: "select", pygame.K_SPACE: "select",
    pygame.K_ESCAPE: "menu", pygame.K_p: "menu",
    pygame.K_r: "defaults", pygame.K_c: "camera",
}
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


def station_label(x: float) -> str:
    if x >= 1.45:
        return "front bumper"
    if x >= 0.72:
        return "front axle / wing"
    if x >= 0.15:
        return "front door"
    if x >= -0.35:
        return "B-pillar (at the CG)"
    if x >= -1.20:
        return "rear door"
    if x >= -1.75:
        return "rear axle / quarter"
    return "rear bumper"


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


@dataclass
class CarBuild:
    """Three slots, a name, and the mirror lock. `cfg_kwargs(lib)` turns it
    into VehicleConfig fields; `hud_kwargs(lib)` into what the renderer
    draws. A build whose flanks carry the SAME published panel and no top
    wing maps onto the closed-form path exactly (that is the study's car)."""

    name: str = "my corsa"
    left: Slot = field(default_factory=lambda: Slot("", 0.97, 0.90, 0.0))
    right: Slot = field(default_factory=lambda: Slot("", 0.97, 0.90, 0.0))
    top: Slot = field(default_factory=lambda: Slot("", -0.90, 1.55, 6.0, "active"))
    mirror: bool = True
    builtin: bool = False

    def slot(self, key: str) -> Slot:
        return getattr(self, key)

    # -- geometry limits -----------------------------------------------------
    def clamp(self, lib: "Library | None" = None) -> "CarBuild":
        for key in ("left", "right"):
            s = self.slot(key)
            c = DEV_CHORD
            if lib is not None and s.wing in lib.wings:
                c = lib.wings[s.wing].chord
            s.x = min(max(s.x, CAR_X_REAR + 0.5 * c), CAR_X_FRONT - 0.5 * c)
            s.h = min(max(s.h, H_W_MIN), H_W_MAX)
            s.inc_deg = min(max(s.inc_deg, INC_MIN), INC_MAX)
            s.mode = "active"
        t = self.top
        t.x = min(max(t.x, TOP_X_MIN), TOP_X_MAX)
        t.h = min(max(t.h, deck_z(t.x) + TOP_STOW_GAP + 0.08), TOP_H_MAX)
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

    def reset(self) -> None:
        self.left = Slot("", 0.97, 0.90, 0.0)
        self.right = Slot("", 0.97, 0.90, 0.0)
        self.top = Slot("", -0.90, 1.55, 6.0, "active")
        self.mirror = True

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
        return out

    def summary(self, lib: "Library") -> str:
        w = self.wings(lib)
        parts = []
        for k in SLOTS:
            s = self.slot(k)
            parts.append(f"{k}: {w[k].name if w[k] else 'none'} x {s.x:+.2f} h {s.h:.2f} inc {s.inc_deg:+.0f}")
        return "   ".join(parts)

    # -- persistence ---------------------------------------------------------
    def to_json(self) -> dict:
        return dict(version=2, name=self.name, mirror=self.mirror, builtin=self.builtin,
                    slots={k: asdict(self.slot(k)) for k in SLOTS})

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
        return cls(name=str(d.get("name", "my corsa")), left=_slot("left", base.left),
                   right=_slot("right", base.right), top=_slot("top", base.top),
                   mirror=bool(d.get("mirror", True)), builtin=bool(d.get("builtin", False)))

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


def _could_not_save(exc: Exception) -> str:
    """The hint for a library / file write that failed (a full disk, a
    read-only runs/, a name whose file another record holds): the garage
    stays up and says so."""
    return f"could not save: {getattr(exc, 'strerror', None) or exc}"


def _dev_aero(spec: "WingSpec | None", slot: Slot):
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
    if spec is None:
        return None
    if "CLa" not in spec.aero or abs(float(spec.aero.get("ride_h") or 0.0) - slot.h) > 1e-6:
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


def build_car_mesh(paint=C_PAINT) -> list[tuple[np.ndarray, tuple, str]]:
    """(verts (n,3), colour, kind). Kind is 'body' | 'wheel' | 'trim'.

    `paint` is the body colour (the player's paint, drive/paint.py); the nose
    and tail caps are a darker tone of it. The stock yellow keeps its own
    hand-picked cap tone, C_PAINT_DARK; any other paint gets 0.75 of itself,
    about what C_PAINT_DARK is of C_PAINT (0.744-0.757 by channel)."""
    paint = tuple(int(c) for c in paint)
    dark = (C_PAINT_DARK if paint == C_PAINT
            else tuple(int(round(0.75 * c)) for c in paint))
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


def wing_polys(spec: "WingSpec | None", key: str, slot: Slot, deploy: float,
               lib: "Library | None", selected: bool = False, legacy_type: str = "") -> list:
    """The polygons of one slot's wing at this deploy fraction.

    Flank: a vertical loft of the section (suction side towards the car,
    since its lift is the inward side force), standing off the sill by
    DEV_OUT0 + DEV_OUT1 * deploy, two struts, optional end plates.
    Top: an inverted loft across the car; stowed it lies on the deck,
    deployed it rises to the slot height and takes its incidence; end
    plates hang towards the road; two pylons."""
    polys = []
    role = SLOT_ROLE[key]
    if spec is None and not legacy_type:
        return polys
    if spec is None:                                   # the published panel, drawn as before
        chord, span, taper, twist, plate, sec_name = DEV_CHORD, DEV_SPAN, 1.0, 0.0, (0.06 if legacy_type == "plate" else 0.0), ("naca6412" if legacy_type == "plate" else "naca4412")
    else:
        chord, span, taper, twist, plate, sec_name = spec.chord, spec.span, spec.taper, spec.twist_deg, spec.plate_h, spec.airfoil
    loop = _section_loop(lib, sec_name)
    on = bool(spec is not None or legacy_type)
    col = C_PANEL_SEL if selected else (C_PANEL_ON if on else C_PANEL_OFF)
    n_st = 6
    if role == "flank":
        s = 1.0 if key == "left" else -1.0
        out = DEV_OUT0 + DEV_OUT1 * deploy
        yc = s * (CAR_HALF_W + out)
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
        for dz in (-0.28 * span, 0.28 * span):
            z = slot.h + dz
            polys += _box(slot.x - 0.015, slot.x + 0.015, min(s * CAR_HALF_W, yc), max(s * CAR_HALF_W, yc),
                          z - 0.012, z + 0.012, C_STRUT)
        if plate > 0.0:
            for sgn in (-1.0, 1.0):
                z = slot.h + sgn * 0.5 * span
                polys += _box(slot.x - 0.6 * chord * taper, slot.x + 0.6 * chord * taper,
                              yc - 0.5 * plate, yc + 0.5 * plate, z - 0.006, z + 0.006, C_PLATE, "plate")
    else:
        deck = deck_z(slot.x)
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
        c_tip = chord * taper
        z_tip = zc - 0.5 * span * math.sin(ang) * 0.0
        if plate > 0.0:
            for sgn in (-1.0, 1.0):
                y = sgn * (0.5 * span + 0.008)
                polys += _box(slot.x - 0.65 * c_tip, slot.x + 0.65 * c_tip, y - 0.006, y + 0.006,
                              z_tip - plate * deploy - 0.02 * (1 - deploy), z_tip + 0.03, C_PLATE, "plate")
        for sgn in (-1.0, 1.0):
            y = sgn * 0.28 * span
            polys += _box(slot.x - 0.15 * chord, slot.x - 0.15 * chord + 0.06, y - 0.012, y + 0.012,
                          deck, zc - 0.02 * chord, C_STRUT)
    return polys


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
        self.reset()
        self.fov_deg = 38.0

    def reset(self) -> None:
        self.yaw = math.radians(38.0)        # from the front-left quarter
        self.pitch = math.radians(19.0)
        self.dist = 7.6
        self.target = np.array([-0.15, 0.0, 0.70])

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
        self.dist = min(max(self.dist * k, 3.2), 16.0)


# =========================================================================== #
#  THE 3-D VIEW (the CAR page)                                                 #
# =========================================================================== #
class GarageView:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.W, self.H = screen.get_size()
        self.ui = min(self.W / 1280.0, self.H / 800.0)
        self.fonts = ui.Fonts(self.ui)
        self.text = ui.Text(self.fonts)
        self.f_lbl = self.fonts.get(14)
        self.f_val = self.fonts.get(16)
        self.f_big = self.fonts.get(26, bold=True)
        self.paint = C_PAINT
        self.car = Batch(build_car_mesh(self.paint))
        self.n_polys = 0
        self.frame_ms = 0.0
        self.show_vectors = True

    def set_paint(self, rgb) -> None:
        """Paint the preview car `rgb` (None: the stock C_PAINT yellow). The
        body Batch is rebuilt only when the colour changes (~2.4 ms); the next
        frame draws it. Cosmetic: nothing in the build or its JSON changes.
        The preview is the old hatch shell whatever car is fitted, so the
        caller passes the fitted car's paint RESOLVED (its factory colour
        for 'factory', render.factory_colour)."""
        rgb = C_PAINT if rgb is None else tuple(int(c) for c in rgb)
        if rgb != self.paint:
            self.paint = rgb
            self.car = Batch(build_car_mesh(rgb))

    def _txt(self, s, x, y, font=None, col=C_TEXT):
        surf = (font or self.f_val).render(s, True, col)
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
            polys += wing_polys(wings[key], key, build.slot(key), deploy, lib, selected=(key == selected))
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
             pad_name: str | None, hint: str = "", status: str = "") -> None:
        t0 = time.perf_counter()
        self.draw_scene(build, lib, cam, deploy, selected)
        if self.show_vectors:
            self._draw_vectors(build, lib, cam, deploy)
        self._draw_dimensions(build, selected, cam)
        self._draw_info(build, lib, selected, deploy)
        self._draw_help(pad_name, hint, status)
        self.frame_ms = (time.perf_counter() - t0) * 1e3

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
            dp = design_point(spec, slot.inc_deg, V=V_REF[spec.role], x_w=slot.x)
            if not dp:
                continue
            if key == "top":
                z_stow = deck_z(slot.x) + TOP_STOW_GAP
                base = np.array([slot.x, 0.0, z_stow + (slot.h - z_stow) * deploy])
                f_dir = np.array([0.0, 0.0, -1.0])
            else:
                s = 1.0 if key == "left" else -1.0
                base = np.array([slot.x, s * (CAR_HALF_W + DEV_OUT0 + DEV_OUT1 * deploy), slot.h])
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
            self._txt(f"vectors at V_REF: {1.0 / VEC_M_PER_N:.0f} N = 1 m   F force   D drag   (V hides)",
                      12 * u, 668 * u, self.f_lbl, C_TEXT_DIM)

    def _draw_ground(self, cam: Orbit) -> None:
        quad = np.array([(-6.0, -5.0, 0.0), (6.0, -5.0, 0.0),
                         (6.0, 5.0, 0.0), (-6.0, 5.0, 0.0)])
        scr, d = cam.project(quad)
        if np.all(d > 0.05):
            pygame.draw.polygon(self.screen, C_GROUND, scr.astype(np.int32).tolist())
        for x in np.arange(-4.0, 4.01, 0.5):
            self._line3(cam, (x, -3.0, 0.0), (x, 3.0, 0.0), C_GRID)
        for y in np.arange(-3.0, 3.01, 0.5):
            self._line3(cam, (-4.0, y, 0.0), (4.0, y, 0.0), C_GRID)
        foot = np.array([(CAR_X_FRONT - 0.05, 0.62, 0.003), (CAR_X_FRONT - 0.35, 0.86, 0.003),
                         (CAR_X_REAR + 0.30, 0.86, 0.003), (CAR_X_REAR, 0.66, 0.003),
                         (CAR_X_REAR, -0.66, 0.003), (CAR_X_REAR + 0.30, -0.86, 0.003),
                         (CAR_X_FRONT - 0.35, -0.86, 0.003), (CAR_X_FRONT - 0.05, -0.62, 0.003)])
        scr, d = cam.project(foot)
        if np.all(d > 0.05):
            pygame.draw.polygon(self.screen, C_SHADOW, scr.astype(np.int32).tolist())

    def _draw_dimensions(self, build: CarBuild, key: str, cam: Orbit) -> None:
        """CG marker, x along the ground, h up the side, for the selected slot."""
        slot = build.slot(key)
        s = -1.0 if key == "right" else 1.0
        y_side = s * (CAR_HALF_W + DEV_OUT0 + 0.45)
        self._line3(cam, (-0.15, 0.0, 0.0), (0.15, 0.0, 0.0), C_DIM, 2)
        self._line3(cam, (0.0, -0.15, 0.0), (0.0, 0.15, 0.0), C_DIM, 2)
        p = cam.project(np.array([(0.0, 0.0, 0.0)]))[0][0]
        self._txt("CG", p[0] + 6, p[1] - 22, self.f_lbl, C_DIM)
        self._line3(cam, (0.0, y_side, 0.0), (slot.x, y_side, 0.0), C_DIM, 2)
        self._line3(cam, (0.0, y_side - 0.1, 0.0), (0.0, y_side + 0.1, 0.0), C_DIM, 2)
        self._line3(cam, (slot.x, y_side - 0.1, 0.0), (slot.x, y_side + 0.1, 0.0), C_DIM, 2)
        p = cam.project(np.array([(0.5 * slot.x, y_side + s * 0.15, 0.0)]))[0][0]
        self._txt(f"x {slot.x:+.2f} m", p[0] - 40, p[1] + 4, self.f_lbl, C_DIM)
        yp = s * (CAR_HALF_W + DEV_OUT0 + 0.5 * DEV_THICK + 0.10) if key != "top" else 0.5 * 1.7 + 0.15
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
            return (len(lines) - 1) * font.get_linesize()

        slot = build.slot(key)
        spec = lib.wings.get(slot.wing)
        on = spec is not None
        self._txt(self._fit(SLOT_LABEL[key] + ("   (mirrored)" if build.mirror and key != "top" else ""),
                            self.f_lbl, wmax), x, y, self.f_lbl, C_PANEL_ON if on else C_TEXT_DIM)
        y += 20 * u
        self._txt(self._fit(spec.name if on else "none", self.f_big, wmax), x, y, self.f_big,
                  C_PANEL_ON if on else C_TEXT_DIM)
        y += 40 * u
        if on:
            src = "XFOIL" if not spec.aero.get("polar_is_estimate", True) else "estimate"
            geo = (f"{spec.airfoil}  b {spec.span:.2f} c {spec.chord:.2f} taper {spec.taper:.2f} "
                   f"S {spec.S:.2f} m2")
            if spec.legacy:
                geo = f"published panel S {spec.legacy.get('S', 0.35):.2f} m2 CL0 {spec.legacy['CL0']:.2f} L/D {spec.legacy['LD']:.1f}"
            rows = [(geo, C_TEXT_DIM)]
        else:
            rows = ([("Start here: W puts a ready-made wing\nin this slot", C_PANEL_ON)]
                    if key != "top" else [])
            rows += [("W  try a ready-made wing", C_TEXT_DIM), ("D  design your own", C_TEXT_DIM)]
        rows += [
            (f"station x    {slot.x:+.2f} m", C_TEXT),
            (f"             {station_label(slot.x)}", C_TEXT_DIM),
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
            V = V_REF[spec.role]
            dp = design_point(spec, slot.inc_deg, V=V, x_w=slot.x)
            if dp:
                tag = "" if (spec.legacy or not spec.aero.get("polar_is_estimate", True)) else "  (ESTIMATE polar)"
                y += put(f"AT {V:.1f} m/s{tag}", y, self.f_lbl, C_TEXT_DIM)
                y += 20 * u
                y += put(f"CL {dp['CL']:.2f}  F {dp['F']:4.0f} N  D {dp['D']:3.0f} N  L/D {dp['LD']:.1f}",
                         y, self.f_val)
                y += 22 * u
                if spec.role == "flank":
                    y += put(f"= {100 * dp['F'] / (CAR.m * G):.2f}% of mg  (x+b)/b x{(slot.x + CAR.b) / CAR.b:.2f}",
                             y, self.f_lbl, C_TEXT_DIM)
                    y += 22 * u
                    y += put("corner-speed gain (crossover.gain)", y, self.f_lbl, C_TEXT_DIM)
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
                    share_f = (slot.x + CAR.b) / CAR.L
                    y += put(f"downforce split  front {100 * share_f:.0f}%  rear {100 * (1 - share_f):.0f}%",
                             y, self.f_lbl, C_OK if share_f > 0.3 else C_WARN)
                    y += 22 * u
                    y += put(f"= {100 * dp['F'] / (CAR.m * G):.2f}% of mg; a front-limited car wants it forward",
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
        self._txt(self._fit(f"preview: {'DEPLOYED' if deploy > 0.5 else 'stowed'}   {build.summary(lib)}",
                            self.f_lbl, wmax), x, y, self.f_lbl,
                  C_PANEL_ON if deploy > 0.5 else C_TEXT_DIM)

    def _draw_help(self, pad_name, hint, status="") -> None:
        u = self.ui
        r = self._panel((12, 690, 1256, 98))
        x, y = r.x + 10 * u, r.y + 6 * u
        lines = [
            ("GARAGE  -  three wings: flank left / right, top.  1 2 3 select, arrows place, W wing, "
             "D design, A airfoils, L library", C_TEXT),
            ("mouse drag orbit | wheel zoom | LEFT/RIGHT x | UP/DOWN h | [ ] incidence | M mirror | "
             "T top mode | SPACE deploy preview | V vectors | R defaults | C camera", C_TEXT_DIM),
            ("ENTER drive  |  ESC: wing tutorial, controls, defaults, quit", C_TEXT_DIM),
        ]
        if pad_name:
            lines[2] = (f"PS5 {pad_name}:  L-stick move | R-stick orbit | L1/R1 incidence | "
                        "TRIANGLE slot | SQUARE wing | CIRCLE deploy | L3 design | CROSS drive | "
                        "OPTIONS menu", C_OK)
            lines.append(("ENTER drive  |  ESC: wing tutorial, controls, defaults, quit", C_TEXT_DIM))
        else:
            lines.append(("no controller: pair the DualSense over Bluetooth and press PS - "
                          "it hot-plugs here and in the drive", C_TEXT_DIM))
        for s, c in lines:
            self._txt(s, x, y, self.f_lbl, c)
            y += 18 * u
        if hint:
            self._txt(hint[:60], r.right - 10 * u - self.f_lbl.size(hint[:60])[0], r.y + 6 * u,
                      self.f_lbl, C_PANEL_ON)
        if status:
            self._txt(status, r.right - 10 * u - self.f_lbl.size(status)[0], r.y + 42 * u,
                      self.f_lbl, C_TEXT_DIM)


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
#  THE DESIGNER PAGE                                                           #
# =========================================================================== #
#: What the wing page maximises. **`lap time` is first, and it is first
#: because it is the only one of these that does not need the designer to
#: state an exchange rate.** The others price drag with a CAP or a FLOOR the
#: designer types; the lap prices it physically, on the straights of the
#: circuit stated in step 1, and prices the force it buys in the corners of
#: the same circuit. They are kept because a cap is still the right question
#: when the cap is a real constraint (a class rule, a mounting limit) rather
#: than a stand-in for a lap nobody had.
OBJECTIVES = {
    "flank": ("lap time", "corner gain @ drag cap", "force / drag @ force floor",
              "max force @ drag cap"),
    "top": ("lap time", "downforce @ drag cap", "Fz / drag @ downforce floor"),
}


class Designer:
    """Edits ONE wing for ONE slot: a working copy of the library wing (or a
    fresh one for an empty / published slot), re-analysed on every edit."""

    def __init__(self, g: "Garage", key: str):
        self.g, self.lib, self.key = g, g.lib, key
        self.role = SLOT_ROLE[key]
        slot = g.build.slot(key)
        base = self.lib.wings.get(slot.wing)
        if base is None or base.legacy:
            seed = self.lib.wings.get("flank-e423" if self.role == "flank" else "rear-s1223")
            if seed is None:
                seed = WingSpec(role=self.role, airfoil="e423" if self.role == "flank" else "s1223",
                                span=0.80 if self.role == "flank" else 1.40,
                                chord=0.45 if self.role == "flank" else 0.30, taper=0.85, plate_h=0.06)
            self.spec = seed.copy(name=self.lib.unique_name("wings", "flank-new" if self.role == "flank" else "rear-new"),
                                  builtin=False, legacy=None)
            self.origin = None
            self.dirty = True
        else:
            self.spec = base.copy()
            self.origin = base.name
            self.dirty = False
        self.objective = OBJECTIVES[self.role][0]
        #  THE MISSION, stated before the section or the planform. AeroBO's
        #  cartrack.py makes the argument: "maximise a downforce coefficient
        #  against CD_budget" is a calibration, not a requirement -- measured
        #  there, the budget admitted 1857 of 1857 feasible draws, so it
        #  decided nothing, while a two-point requirement pair admitted 5.
        #  The speed and the pair below ARE that requirement, and they are the
        #  first three rows of the page for the same reason they are the first
        #  thing AeroBO asks: a section and a planform cannot be judged until
        #  someone says what the wing is for.
        #  The mission is the garage's, stated in step 1, and the wing is
        #  scored against the SAME one the section was -- which is the whole
        #  reason it is held there and not here.
        self.profile = g.mission.profile(make_track)
        self.mu_scale = g.mission.mu_scale
        self.base = g.build.mission_aero(g.lib, exclude=key)
        self.base_lap = ms.lap(self.profile, self.base, mu_scale=self.mu_scale)
        #  the design speed is no longer a module constant OR a typed row: it
        #  is the lap's own time-weighted mean speed (length / time), which is
        #  where the section's Reynolds number and every coefficient on this
        #  page are read. Still editable -- a designer who wants the panel
        #  read at its limit corner may say so.
        self.V_design = self.base_lap.v_mean if self.base_lap.ok else V_REF[self.role]
        self.drag_cap = 80.0 if self.role == "flank" else 60.0
        self.force_floor = 150.0 if self.role == "flank" else 300.0
        #  the wing's budget is the same measured law, at the wing's own row
        #  count -- 7 rows fixed-area, 8 with the area freed.
        self.effort = "balanced"
        self.budget = opt.budget_for(len(design_vars(self.role)), self.effort)
        #  AeroBO's "keep going": how many MORE evaluations a continuation
        #  buys once the budget is spent. Sized like a quick top-up, and
        #  editable on the CONVERGENCE step
        self.more = 8
        #  IS THE REFERENCE AREA A DESIGN VARIABLE? AeroBO's `CarWingProblem`
        #  asks exactly this (`area_bounds_m2`), and `wing.design_table` has
        #  carried the row since it was ported -- it was simply never on a
        #  form. With the area FIXED the span row IS the aspect ratio and
        #  every candidate is compared on one reference, which is what makes a
        #  coefficient objective meaningful. Free it and two candidates no
        #  longer share a reference, so the score has to be read in FORCES --
        #  the page says so rather than letting a coefficient quietly change
        #  what it is quoted against.
        self.area_free = False
        #  THE DESIGN BOX: the band each design variable is SEARCHED over,
        #  which is not the same question as the value it currently holds.
        #  It opens on `wing.BOUNDS` -- the packaging bands -- and a row
        #  narrowed here narrows the search and nothing else. `design_bounds`
        #  has always taken per-variable overrides; until now nothing handed
        #  it any but the span's.
        self.box = {k: [float(lo), float(hi)]
                    for k, (lo, hi) in BOUNDS[self.role].items()}
        self.lap = None
        self.result = None
        self.msg = ""
        self.polar = None
        self.dp = {}
        self.span_data = None
        self.err = ""
        self.params = ui.ParamList(self._build_params(), title=f"DESIGN  {SLOT_LABEL[key]}")
        self.update()

    # -- the parameter rows --------------------------------------------------
    def _set(self, attr, lo=None, hi=None, cap=None):
        """`cap()` is a band that moves while the page is open -- the flank
        panel's span depends on the slot height, so `Param.hi` (read once at
        build time) cannot express it and the setter has to. Without it the
        page offered spans the optimiser was forbidden to propose."""
        def f(v):
            if cap is not None:
                lo_c, hi_c = cap()
                v = min(max(float(v), lo_c), hi_c)
            setattr(self.spec, attr, float(v) if lo is not None else v)
            self.spec.clamp()
            self.dirty = True
            self.update()
        return f

    def _set_mount(self, v):
        """The mount is a discrete choice like the section, so it cycles
        rather than steps -- and it changes the LATTICE (an endplate mount
        forces its structural plate height), so the wing has to be re-analysed
        exactly as a planform change does."""
        self.spec.mount = str(v) if str(v) in MOUNTS else "pylon"
        self.spec.clamp()
        self.dirty = True
        self.update()

    def _set_blend(self, v):
        """How the wing and the plate MEET (`aero.blend`): the fraction of the
        plate's arc spent turning out of the wing plane. It changes the
        LATTICE -- the plate's line, its section and its toe all ramp on the
        turn instead of stepping at the junction -- so the wing is re-analysed
        the way a planform change is.

        Raising it off zero switches the JUNCTION CHARGE on, which is AeroBO's
        own behaviour ("the junction charge defaults ON with the blend"). The
        credit for softening the corner IS the reason to soften it, and a
        blend priced with the lattice alone is all cost and no benefit: the
        lattice cannot see a corner. Dropping back to zero leaves the charge
        where the user put it -- it is its own row."""
        v = min(max(float(v), 0.0), 1.0)
        was = self.spec.plate_blend
        self.spec.plate_blend = v
        if v > 0.0 and was <= 0.0:
            self.spec.plate_junction = True
        self.spec.clamp()
        self.dirty = True
        self.update()

    def _set_choice(self, attr, allowed, fallback):
        """A discrete lattice choice: cycle it, clamp it, re-analyse."""
        def f(v):
            val = str(v)
            setattr(self.spec, attr, val if val in allowed else fallback)
            self.spec.clamp()
            self.dirty = True
            self.update()
        return f

    def _set_junction(self, v):
        self.spec.plate_junction = (str(v) == "charged")
        self.dirty = True
        self.update()

    def _span_band(self) -> tuple[float, float]:
        """The span band BOTH the page's row and the optimiser use, at the
        slot height the page is showing (`wing.span_fit`). One function, so
        the page cannot offer a panel the optimiser may not propose -- which
        it did: at h = 1.15 m the row went to 1.05 m and the optimiser
        stopped at 0.40 m."""
        lo = BOUNDS[self.role]["span"][0]
        return lo, span_fit(self.role, self.g.build.slot(self.key).h)

    def _set_ride(self, v):
        """The ride-height row. For a TOP wing the slot's height IS the gap to
        the track, so the row writes both and the slot keeps no second opinion;
        for a FLANK panel it is the deployed standoff to the car's own side,
        which is the wing's alone -- the slot's h is a packaging number there
        and the image plane never sees it."""
        lo, hi = BOUNDS[self.role]["ride_h"]
        v = min(max(float(v), lo), hi)
        self.spec.ride_h = v
        self.spec.clamp()
        if self.role == "top":
            slot = self.g.build.slot(self.key)
            slot.h = v
            self.g.build.clamp(self.lib)
            self.g.build.sync_mirror(self.key)
        self.dirty = True
        self.update()

    def _slot_set(self, attr):
        def f(v):
            slot = self.g.build.slot(self.key)
            setattr(slot, attr, v)
            self.g.build.clamp(self.lib)
            self.g.build.sync_mirror(self.key)
            self.update()
        return f

    def _build_params(self):
        b = BOUNDS[self.role]
        slot = self.g.build.slot(self.key)
        names = sorted(self.lib.airfoils)
        P = ui.Param
        rows = [
            #  MISSION -> SECTION -> PLANFORM, AeroBO's order: the mission
            #  layer states the task, the section is designed in 2-D where a
            #  candidate costs milliseconds, and the wing is asked afterwards
            #  whether the winner helped it.
            P("ms", "MISSION  (stated in step 1)", None, kind="label"),
            P("trk", "circuit", lambda: f"{self.g.mission.track} / {self.g.mission.surface}",
              None, kind="choice", choices=[], enabled=False,
              help="ESC back twice to change it. The section was designed against "
                   "this one, so changing it here would make the two stages "
                   "incomparable -- which is why it is not a row"),
            P("blap", "lap without this wing", lambda: self.base_lap.time, None,
              lo=None, hi=None, unit="s", fmt="{:.3f}", enabled=False,
              help="the car as it stands with this slot EMPTY. Every lap number on "
                   "this page is quoted against it"),
            P("vdes", "design speed", lambda: self.V_design, self._set_attr("V_design"),
              step=1.0, fine=0.25, lo=10.0, hi=80.0, unit="m/s", fmt="{:.1f}",
              help="the lap's own time-weighted mean speed (length / time). Every "
                   "coefficient on this page is read at it, and so is the section's "
                   "Reynolds number"),
            P("floor", "force floor", lambda: self.force_floor, self._set_attr("force_floor"),
              step=10.0, fine=2.0, lo=10.0, hi=2000.0, unit="N", fmt="{:.0f}",
              help="read ONLY by the force objectives. The lap does not need it: it "
                   "prices the force in its own corners"),
            P("cap", "drag cap", lambda: self.drag_cap, self._set_attr("drag_cap"),
              step=5.0, fine=1.0, lo=5.0, hi=400.0, unit="N", fmt="{:.0f}",
              help="read ONLY by the capped objectives. The lap does not need it "
                   "either -- it charges every newton of drag on the straights"),
            P("sec", "SECTION", None, kind="label"),
            P("airfoil", "airfoil", lambda: self.spec.airfoil, self._set("airfoil"), kind="choice",
              choices=names, help="LEFT/RIGHT cycles the section library; A opens it with the polar plots"),
            P("browse", "browse the airfoil library  (A)", None, lambda _: self.g.open_airfoils(), kind="action"),
            #  THE DESIGN VECTOR, in AeroBO's `evaluate_car_wing` order --
            #  taper, root twist, tip twist, incidence, end plates, ride, span.
            #  The order is not cosmetic: `wing.design_table` is the one table
            #  the optimiser's bounds, start vector, decode and write-back all
            #  walk, so the page and the search cannot drift apart. The slot's
            #  INCIDENCE sits in the middle of the planform rows because that
            #  is where AeroBO's `alpha_deg` sits; it is still the slot's.
            P("pf", "PLANFORM  (the design vector, in order)", None, kind="label"),
            P("taper", "taper (tip/root)", lambda: self.spec.taper, self._set("taper", *b["taper"]),
              step=0.05, fine=0.01, lo=b["taper"][0], hi=b["taper"][1], fmt="{:.2f}"),
            P("twistr", "root twist", lambda: self.spec.twist_root_deg,
              self._set("twist_root_deg", *b["twist_root_deg"]),
              step=0.5, fine=0.1, lo=b["twist_root_deg"][0], hi=b["twist_root_deg"][1],
              unit="deg", fmt="{:+.1f}",
              help="the second twist row AeroBO carries: the lattice's twist is "
                   "linear root -> tip, so root 0 is the single-row wing"),
            P("twist", "tip twist", lambda: self.spec.twist_deg, self._set("twist_deg", *b["twist_deg"]),
              step=0.5, fine=0.1, lo=b["twist_deg"][0], hi=b["twist_deg"][1], unit="deg", fmt="{:+.1f}",
              help="washout (tip below root) unloads the tip: e up, stall margin up"),
            P("inc", "incidence", lambda: self.g.build.slot(self.key).inc_deg, self._slot_set("inc_deg"),
              step=0.5, fine=0.1, lo=b["inc_deg"][0], hi=b["inc_deg"][1], unit="deg", fmt="{:+.1f}",
              help="AeroBO's alpha row: the angle the wing is bolted on at. It belongs "
                   "to the SLOT, not the wing, and the optimiser moves it too"),
            P("plate", "end plates", lambda: self.spec.plate_h, self._set("plate_h", *b["plate_h"]),
              step=0.01, fine=0.002, lo=b["plate_h"][0], hi=b["plate_h"][1], unit="m",
              help="tip plates in the lattice: cut induced drag, add wetted area"),
            P("ride", "ride height" if self.role == "top" else "standoff",
              lambda: self.spec.ride_h_flown, self._set_ride,
              step=0.05, fine=0.01, lo=b["ride_h"][0], hi=b["ride_h"][1], unit="m",
              help="AeroBO's ride_height row: the gap to the wall the wing is imaged in, "
                   "which is what ground effect is a function of. The TRACK for a top wing; "
                   "the car's own flank for a flank panel, where it is the deployed standoff"),
            P("span", "span" if self.role == "top" else "span (vertical)", lambda: self.spec.span,
              self._set("span", *b["span"], cap=self._span_band),
              step=0.02, fine=0.005, lo=b["span"][0], hi=b["span"][1], unit="m",
              help="the lattice's first-order variable: at fixed area span IS aspect ratio; "
                   "a flank panel is also capped by the sill/roof fit at this slot height"),
            #  DERIVED, not a row. AeroBO sizes by area and span and lets the
            #  chord fall out; carrying a free chord alongside both would let a
            #  candidate be scored against an area it does not have.
            P("chord", "root chord  = 2S / b(1+taper)", lambda: self.spec.chord, None,
              lo=None, hi=None, unit="m", enabled=False,
              help="derived from the reference area, the span and the taper -- "
                   "move the span and watch it follow"),
            P("area", "reference area S", lambda: self.spec.S, None,
              lo=None, hi=None, unit="m2", fmt="{:.3f}", enabled=False,
              help="fixed: the coefficients beside it are quoted against it, so two "
                   "candidates share a reference. AeroBO makes it a row only when a "
                   "band is declared, and then the score has to be read in forces"),
            P("mount", "mount", lambda: self.spec.mount, self._set_mount, kind="choice",
              choices=list(MOUNTS_BUILDABLE),
              help="pylon: two struts in the flow (mount drag + junction interference).  "
                   "endplate: carried by its tip plates instead -- no struts, plates forced "
                   f"to {MOUNT_PLATE_H.get(self.role, 0.06):.2f} m, less tip loss.  "
                   "Two layouts, as AeroBO's own car wing has: 'none' is an "
                   "idealisation and a legacy parity setting, not a way to hold a "
                   "wing up, so it is not offered here"),
            #  HOW THE WING AND THE PLATE MEET. A setting, not a design
            #  coordinate -- AeroBO carries its own blend the same way
            #  (`replace(prob, blend_frac=...)`, never a row of X), because it
            #  decides WHAT geometry is being searched rather than where
            #  inside it to look. Same reason the mount and the area switch
            #  live here.
            P("blend", "plate blend", lambda: self.spec.plate_blend, self._set_blend,
              step=0.05, fine=0.01, lo=0.0, hi=1.0, fmt="{:.2f}",
              enabled=lambda: self.spec.plate_h_flown > 0.0,
              help="0 = the plate is bolted on at a right angle and the surface changes "
                   "from wing to plate -- its line, its section and its twist -- in ONE "
                   "STEP at the junction. Raise it and the plate leaves the wing "
                   "TANGENTIALLY and becomes itself over that fraction of its arc. It "
                   "keeps its arc, so it trades tip height for outboard reach, and the "
                   "wing pays for that reach out of its own span"),
            P("bshape", "blend shape", lambda: self.spec.plate_shape,
              self._set_choice("plate_shape", bl.BLEND_SHAPES, "arc"),
              kind="choice", choices=list(bl.BLEND_SHAPES),
              enabled=lambda: self.spec.plate_h_flown > 0.0 and self.spec.plate_blend > 0.0,
              help="which turn law draws the corner. arc: constant radius -- a circular "
                   "fillet, whose curvature JUMPS at both ends (a crease each side).  "
                   "smooth / spiral: curvature vanishes at both ends, so the surfaces "
                   "meet crease-free; the clothoid 'spiral' buys that with an 11 % "
                   "gentler elbow than the smoothstep. All three are the same arc, the "
                   "same cant and the same wetted area"),
            P("junc", "junction interference",
              lambda: ("charged" if self.spec.plate_junction else "not charged"),
              self._set_junction, kind="choice", choices=["not charged", "charged"],
              enabled=lambda: self.spec.plate_h_flown > 0.0,
              help="the lattice values a corner only through the wake line it draws, so "
                   "the interference drag of two surfaces meeting at an angle is "
                   "INVISIBLE to it -- and that drag is the whole reason to blend. This "
                   "charges Hoerner's correlation at the two corners with a fillet "
                   "credit for the blend. OFF by default: the credit is a calibrated "
                   "shape, not a measurement, and a wing analysed without it would move. "
                   "Raising the blend off zero switches it on"),
            #  WHO DECIDES THE SIZE. AeroBO's planform menu asks this and
            #  carsim never did, although `wing.design_table` has carried the
            #  row from the start. It is a TYPE choice and not a box row
            #  because it changes the LENGTH of the design vector.
            P("sized", "reference area",
              lambda: ("searched" if self.area_free else "fixed"),
              self._set_area_free, kind="choice", choices=["fixed", "searched"],
              help="FIXED: the span row IS the aspect ratio and every candidate is "
                   "compared on one reference, which is what makes a coefficient "
                   "mean anything.  SEARCHED: one more row, immediately ahead of the "
                   "span -- and two candidates no longer share a reference, so the "
                   "score has to be read in FORCES, not in CZ"),
            P("mt", "MOUNT (this slot)", None, kind="label"),
            P("x", "station x", lambda: self.g.build.slot(self.key).x, self._slot_set("x"),
              step=0.05, fine=0.01, lo=CAR_X_REAR, hi=CAR_X_FRONT, unit="m", fmt="{:+.2f}",
              help="forward of the CG: (x + b)/b multiplies the flank gain; a top wing behind the rear axle unloads the front"),
        ]
        if self.role == "flank":
            #  the top wing's height IS its ride height and is edited by that
            #  row above; the flank's h is a packaging number (the roll arm and
            #  the sill/roof span fit), which the image plane never sees.
            rows.append(
                P("h", "height h", lambda: self.g.build.slot(self.key).h, self._slot_set("h"),
                  step=0.05, fine=0.01, lo=H_W_MIN, hi=H_W_MAX, unit="m",
                  help="the flank panel's roll arm, and what its span fit is measured from"))
        if self.role == "top":
            rows.append(P("mode", "deploys", lambda: self.g.build.slot(self.key).mode, self._slot_set("mode"),
                          kind="choice", choices=["active", "fixed"],
                          help="active = out under brake or steering, stowed on the straights"))
        rows += [
            P("op", "OPTIMISER (GP Bayesian, AeroBO)", None, kind="label"),
            P("obj", "objective", lambda: self.objective, self._set_attr("objective"), kind="choice",
              choices=list(OBJECTIVES[self.role]),
              help="scored against the MISSION rows at the top of the page"),
            P("effort", "effort", lambda: self.effort, self._set_effort,
              kind="choice", choices=list(opt.EFFORTS),
              help="how much of the reachable improvement the budget is sized for: "
                   "quick 90 %, balanced 95 %, thorough 99 %. AeroBO's three settings "
                   "and its measured law behind them"),
            P("budget", "evaluations", lambda: self.budget, self._set_attr("budget"),
              kind="int", lo=8, hi=160,
              help="AeroBO's measured law, evals = 9.61 + 3.08 d at 95 %, fitted over "
                   "13 cases with a 10.7-evaluation residual RMS. It is a sizing rule, "
                   "not a prediction, which is why it is editable"),
            P("split", "Sobol start / BO",
              lambda: "%d + %d" % opt.split_for(len(design_vars(self.role, area=self.area_free)),
                                                self.budget), None,
              kind="choice", choices=[], enabled=False,
              help="0.5 x d initial points, clamped to [4, 16] -- AeroBO's measured "
                   "seed rule, which beats 1 x d, 2 x d and 4 x d over 15 of its cases"),
            P("run", "run the optimiser  (O)", None, lambda _: self.optimise(), kind="action"),
            P("more", "more evaluations", lambda: self.more, self._set_attr("more"),
              kind="int", lo=1, hi=160,
              help="what 'keep going' buys: the run so far is the GP's training "
                   "set, nothing is re-flown, and the trace carries on from where "
                   "it stopped -- AeroBO's continuation"),
            P("go", "keep going  (K)", None, lambda _: self.optimise(extend=True), kind="action",
              help="continue the last run by 'more evaluations'. Runs from scratch "
                   "if there is no run to continue, or the box has moved under it"),
            P("sv", "SAVE", None, kind="label"),
            P("name", f"name: {self.spec.name}", None, lambda _: self.g.prompt_rename(), kind="action"),
            P("save", "save to the library + use in this slot  (S)", None, lambda _: self.commit(), kind="action"),
            P("xf", "XFOIL polar for this section  (X)", None, lambda _: self.request_xfoil(), kind="action",
              enabled=self.lib.use_xfoil),
        ]
        return rows

    def _set_attr(self, attr):
        def f(v):
            setattr(self, attr, v)
        return f

    def _set_effort(self, v) -> None:
        self.effort = str(v) if str(v) in opt.EFFORTS else "balanced"
        self.budget = opt.budget_for(
            len(design_vars(self.role, area=self.area_free)), self.effort)

    def _set_box(self, attr: str, end: int):
        """One end of one design-box row. The band is held inside the
        PACKAGING band (`wing.BOUNDS`) and non-degenerate: a user may narrow
        the search, never widen it past what the car can take, and never
        invert it -- an inverted band reaches `optimize.maximise` as a box
        with no interior and comes back as 'every candidate was refused',
        which is a true sentence about the wrong thing."""
        lo_p, hi_p = BOUNDS[self.role][attr]
        pad = 1e-4 if attr not in ("taper",) else 1e-3

        def f(v):
            v = min(max(float(v), lo_p), hi_p)
            row = self.box[attr]
            if end == 0:
                row[0] = min(v, row[1] - pad)
            else:
                row[1] = max(v, row[0] + pad)
        return f

    def _get_box(self, attr: str, end: int):
        def f():
            #  the SPAN's upper end is not the user's alone: the sill/roof fit
            #  at this slot height caps it, and the row shows the cap it will
            #  actually be searched under rather than the number typed.
            lo, hi = self.box[attr]
            if attr == "span":
                b_lo, b_hi = self._span_band()
                return max(lo, b_lo) if end == 0 else min(hi, b_hi)
            return lo if end == 0 else hi
        return f

    def _set_area_free(self, v):
        self.area_free = (str(v) == "searched")
        #  the design vector just grew a row, and the budget is a function of
        #  how many rows there are
        self.budget = opt.budget_for(
            len(design_vars(self.role, area=self.area_free)), self.effort)
        #  the row count of the design vector just changed, so anything
        #  holding a vector of the old length is stale
        self.result = None

    def search_bounds(self):
        """`(bounds, labels)` the optimiser is actually given: the design box
        as edited, with the span band intersected with the sill/roof fit.
        ONE function, so the page's band rows and the search cannot disagree
        -- which is the same trap `_span_band` was written for, one level up."""
        names = design_vars(self.role, area=self.area_free)
        over = {}
        for a in names:
            lo, hi = self.box[a]
            if a == "span":
                b_lo, b_hi = self._span_band()
                lo, hi = max(lo, b_lo), min(hi, b_hi)
                if hi <= lo:
                    lo, hi = b_lo, b_hi
            over[a] = (lo, hi)
        return (design_bounds(self.role, area=self.area_free, bands=over),
                design_labels(self.role, area=self.area_free))

    def _box_rows(self):
        """The BAND rows -- two per design variable, in the vector's order.

        AeroBO's stage 3 draws exactly this table and calls it the design box;
        carsim showed the design vector's VALUES under that heading and kept
        the bands as module constants nobody could reach. They are different
        questions: the value is the wing as it stands, the band is what the
        search may propose."""
        P = ui.Param
        rows = [P("bx", "THE BANDS  (what the search may propose)", None, kind="label")]
        for attr, _owner, label, unit, *_b in design_table(self.role, self.area_free):
            lo_p, hi_p = BOUNDS[self.role][attr]
            step = 0.05 if unit == "m2" else (0.5 if unit == "deg" else 0.02)
            fmt = "{:+.1f}" if unit == "deg" else "{:.3f}"
            for end, tag in ((0, "min"), (1, "max")):
                #  the span's ends carry a star: they are the only band the
                #  page does not own outright -- `span_fit` narrows them at
                #  this slot height, and the row SHOWS the narrowed number
                mark = " *" if attr == "span" else ""
                rows.append(P(f"bx.{attr}.{tag}", f"{label}  {tag}{mark}",
                              self._get_box(attr, end), self._set_box(attr, end),
                              step=step, fine=step / 5.0, lo=lo_p, hi=hi_p,
                              unit=unit, fmt=fmt,
                              help=f"the {tag} of the band `optimize.maximise` searches "
                                   f"{label} over. The packaging band is "
                                   f"{lo_p:g} to {hi_p:g} {unit} and this row may only "
                                   f"narrow it"
                                   + ("   (*) and the sill/roof fit at this slot "
                                      "height narrows it again -- the value shown is "
                                      "the one the search will actually get"
                                      if attr == "span" else "")))
        return rows

    # -- analysis --------------------------------------------------------------
    @property
    def slot(self) -> Slot:
        return self.g.build.slot(self.key)

    def update(self) -> None:
        slot = self.slot
        ride = slot.h if self.role == "top" else None
        aero = self.lib.analyse_wing(self.spec, ride_h=ride, V=self.V_design)
        self.err = aero.get("error", "")
        self.polar = self.lib.wing_polar(self.spec)
        self.dp = design_point(self.spec, slot.inc_deg, x_w=slot.x) if "CLa" in aero else {}
        try:
            self.span_data = spanwise(self.spec, self.polar, slot.inc_deg, ride_h=ride)
        except ValueError:
            self.span_data = None
        #  the lap this wing actually does, kept current with every edit so the
        #  read-out moves as a row moves -- 26 ms, the same order as the
        #  lattice solve above it.
        self.lap = self.lap_of(aero, slot.inc_deg) if "CLa" in aero else None
        for p in self.params.params:
            if p.key == "name":
                p.label = f"name: {self.spec.name}" + ("  *" if self.dirty else "")

    def request_xfoil(self) -> None:
        if not self.lib.use_xfoil:
            self.msg = "XFOIL is not on this machine: the estimate polar is used"
            return
        self.lib.polar(self.spec.airfoil, self.spec.reynolds(), want_xfoil=True)
        self.msg = f"XFOIL queued: {self.spec.airfoil} at Re {re_bank_snap(self.spec.reynolds()):.2g}"

    def commit(self, name: str | None = None) -> str:
        """Save the working wing under its name and put it in the slot."""
        if name:
            self.spec.name = name
        spec = self.spec
        #  a built-in keeps its name, and so does another wing whose FILE this
        #  name folds onto ('Flank-E423' is flank-e423.json): saved beside it
        if (spec.name in self.lib.wings and self.lib.wings[spec.name].builtin) or (
                spec.name not in self.lib.wings
                and self.lib.unique_name("wings", spec.name) != spec.name):
            spec.name = self.lib.unique_name("wings", spec.name)
        spec.builtin = False
        spec.legacy = None
        try:
            self.lib.save_wing(spec.copy())
        except (OSError, ValueError) as exc:
            self.msg = _could_not_save(exc)
            return ""
        slot = self.slot
        slot.wing = spec.name
        self.g.build.sync_mirror(self.key)
        self.g.build.clamp(self.lib)
        self.origin = spec.name
        self.dirty = False
        self.update()
        self.msg = f"saved '{spec.name}' to the library and put it in the {self.key} slot"
        return spec.name

    # -- the optimiser --------------------------------------------------------
    def lap_of(self, aero: dict, inc_deg: float) -> "ms.LapResult":
        """This wing, on this car, over the stated circuit."""
        slot = self.slot
        m = ms.merge_wing(self.base, aero, self.role, inc_deg, slot.x, slot.h, slot.mode)
        return ms.lap(self.profile, m, mu_scale=self.mu_scale)

    def _objective_value(self, dp: dict, aero: dict | None = None,
                         inc_deg: float | None = None) -> float:
        F, D = dp["F"], dp["D"]
        obj = self.objective
        if obj == "lap time":
            if aero is None or inc_deg is None:
                return -math.inf
            r = self.lap_of(aero, inc_deg)
            return -r.time if r.ok else -math.inf
        if obj.startswith("corner gain"):
            g = dp.get("gain_pct")
            if g is None or D > self.drag_cap:
                return -math.inf
            return float(g)
        if obj.startswith("force / drag") or obj.startswith("Fz / drag"):
            if F < self.force_floor or D <= 1e-6:
                return -math.inf
            return F / D
        if D > self.drag_cap:
            return -math.inf
        return F

    def optimise(self, extend: bool = False) -> dict | None:
        """GP-BO over the design box. `extend=True` is AeroBO's "keep going":
        continue the last run by `self.more` evaluations instead of starting
        a fresh one (see the block below for when that is allowed)."""
        slot = self.slot
        role = self.role
        b = BOUNDS[role]
        ride = slot.h if role == "top" else None
        polar = self.polar
        if polar is None:
            return None
        from .aero.wing import analyse as _analyse
        #  THE END PLATE'S SECTION HAS TO REACH THE SEARCH. `update` goes
        #  through `library.analyse_wing`, which reads `spec.plate_airfoil`;
        #  this inner loop calls `analyse` directly and so saw a FLAT plate
        #  however the plates had been designed. The wing was then optimised
        #  against a lattice the page was not showing.
        plate_pol = None
        if self.spec.plate_airfoil and self.spec.plate_airfoil in self.lib.airfoils:
            plate_pol = self.lib.polar(self.spec.plate_airfoil,
                                       self.spec.reynolds(self.V_design) * max(self.spec.taper, 0.05),
                                       want_xfoil=False)
        #  The design vector is `wing.DESIGN_VARS` = this page's row order, and
        #  the bounds, the start vector, the objective's decode and the
        #  write-back all walk that one table (`design_bounds` / `design_x0` /
        #  `apply_design`). They used to be four hand-written lists that
        #  happened to agree; now they cannot drift. The span band is the same
        #  sill/roof fit the span row shows, through the same `span_fit`.
        bounds, labels = self.search_bounds()
        #  THE START VECTOR HAS TO BE INSIDE THE BOX THE SEARCH IS GIVEN.
        #  `design_x0` reports the wing AS IT STANDS, and the span band is
        #  narrowed at run time by the sill/roof fit at this slot height
        #  (`_span_band`), so a wing carrying a span the slot cannot take
        #  produced a start vector OUTSIDE the bounds. `optimize.maximise`
        #  clips it before evaluating (so the BO was fine), but `f0` below was
        #  computed on the UNCLIPPED vector -- a score for a wing the search
        #  was forbidden to propose. The comparison `f_best >= f0` then failed
        #  for a packaging reason and the page reported "no feasible design",
        #  which is not what had happened. Measured on the seeded library at
        #  h = 0.90 m: span 1.05 m against a band that stops at 0.88 m.
        b = np.asarray(bounds, dtype=float)
        x0 = list(np.clip(np.asarray(design_x0(self.spec, slot.inc_deg,
                                               area=self.area_free), dtype=float),
                          b[:, 0], b[:, 1]))
        V = self.V_design          # the MISSION row, not a module constant

        def f(x):
            sp = self.spec.copy()
            inc = apply_design(sp, x, clamp=False, area=self.area_free)
            if role == "flank" and (slot.x + 0.5 * sp.chord > CAR_X_FRONT or slot.x - 0.5 * sp.chord < CAR_X_REAR):
                return -math.inf
            try:
                sp.aero = _analyse(sp, polar, V=V, ride_h=ride, plate_polar=plate_pol)
            except ValueError:
                return -math.inf
            dp = design_point(sp, inc, V=V, x_w=slot.x)
            if not dp or dp.get("stalled") or dp["stall_margin_deg"] < 2.0:
                return -math.inf
            return self._objective_value(dp, sp.aero, inc)

        #  A CONTINUATION CONTINUES (AeroBO's "keep going"): the last run's
        #  evaluations are the GP's training set, nothing is re-flown, and
        #  `more` evaluations are bought on top. It is only a continuation of
        #  the SAME search: the box, the objective and the vector have to be
        #  the ones the record was flown on, or the old scores would train a
        #  GP on a problem that no longer exists -- then it starts over.
        prev = self.result if extend else None
        bl = [list(map(float, r)) for r in bounds]
        same = bool(prev is not None and "X" in prev
                    and prev.get("objective") == self.objective
                    and prev.get("area_free") == bool(self.area_free)
                    and prev.get("bounds") == bl
                    and len(prev["X"]) and len(prev["X"][0]) == len(x0))
        t0 = time.perf_counter()
        if same:
            n_more = max(1, int(self.more))
            bo = opt.maximise(f, bounds, n_iter=n_more, seed=0, labels=labels,
                              resume=dict(X=prev["X"], y=prev["y"]))
            rs = opt.random_search(f, bounds, n=n_more, seed=1, labels=labels,
                                   resume=dict(X=prev["rs_X"], y=prev["rs_y"]))
            n = int(bo["n_eval"])
            f0 = float(prev["start"])
            dt = time.perf_counter() - t0 + float(prev.get("secs", 0.0))
        else:
            n = int(self.budget)
            n_init, n_iter = opt.split_for(len(x0), n)
            bo = opt.maximise(f, bounds, n_init=n_init, n_iter=n_iter, seed=0,
                              x0=x0, labels=labels)
            rs = opt.random_search(f, bounds, n=n, seed=1, labels=labels)
            dt = time.perf_counter() - t0
            f0 = f(np.asarray(x0))
        self.result = dict(bo=bo["f_best"], rs=rs["f_best"], start=f0, trace=bo["best_trace"],
                           rs_trace=rs["best_trace"], n=n, secs=dt, objective=self.objective,
                           n_prior=int(bo.get("n_prior", 0)), continued=same,
                           x=bo["x_best"].tolist(), labels=labels,
                           design=format_design(bo["x_best"], role, area=self.area_free),
                           bounds=bl, area_free=bool(self.area_free),
                           #  the observations, so the next "keep going" can
                           #  inherit them rather than re-fly them
                           X=np.asarray(bo["X"], float).tolist(), y=[float(v) for v in bo["y"]],
                           rs_X=np.asarray(rs["X"], float).tolist(), rs_y=[float(v) for v in rs["y"]])
        if math.isfinite(bo["f_best"]) and bo["f_best"] >= f0:
            slot.inc_deg = apply_design(self.spec, bo["x_best"], area=self.area_free)
            self.g.build.sync_mirror(self.key)
            self.dirty = True
            fmt = ((lambda v: f"{-v:.4f} s") if self.objective == "lap time"
                   else (lambda v: f"{v:.3g}"))
            tag = f"continued {bo['n_prior']} -> {n}" if same else f"{n} evals"
            self.msg = (f"BO {fmt(bo['f_best'])} vs random {fmt(rs['f_best'])} vs start {fmt(f0)} "
                        f"({tag}, {dt:.1f} s) - applied")
        else:
            fmt2 = ((lambda v: f"{-v:.4f} s") if self.objective == "lap time"
                    else (lambda v: f"{v:.3g}"))
            if not math.isfinite(bo["f_best"]):
                self.msg = (f"every candidate was refused ({n} evals): the packaging bands, "
                            f"the stall margin or the lap left nothing feasible")
            else:
                self.msg = (f"nothing beat the wing as it stands ({n} evals, {dt:.1f} s): "
                            f"BO {fmt2(bo['f_best'])} vs start {fmt2(f0)}")
        self.update()
        return self.result

    # -- drawing ----------------------------------------------------------------
    def draw(self, screen, text: ui.Text, plot: ui.Plot, u: float) -> str:
        R = lambda x, y, w, h: (int(x * u), int(y * u), int(w * u), int(h * u))   # noqa: E731
        help_ = self.params.draw(screen, text, ui.panel(screen, R(12, 12, 424, 664)), row_h=int(21 * u), size=13)
        spec, slot = self.spec, self.slot
        # -- planform / side view
        r1 = ui.panel(screen, R(444, 12, 404, 226))
        b2, c0, lam = 0.5 * spec.span, spec.chord, spec.taper
        etas = np.linspace(-1, 1, 21)
        cs = c0 * (1 - (1 - lam) * np.abs(etas))
        if self.role == "flank":
            plot.begin(screen, r1, (-0.6 * c0 - 0.1, 0.6 * c0 + 0.1), (-b2 - 0.1, b2 + 0.1),
                       title="side view (x forward, z up)", equal=True)
            xs = np.concatenate([0.5 * cs, -0.5 * cs[::-1]])
            ys = np.concatenate([etas * b2, (etas * b2)[::-1]])
            plot.fill(xs, ys, (70, 45, 20))
            plot.line(xs, ys, C_PANEL_ON, 2, closed=True)
            if spec.plate_h > 0:
                for sgn in (-1, 1):
                    plot.line([-0.6 * c0 * lam, 0.6 * c0 * lam], [sgn * b2, sgn * b2], C_PLATE, 3)
            plot.hline(0.0, ui.C_GRID)
        else:
            plot.begin(screen, r1, (-b2 - 0.1, b2 + 0.1), (-0.6 * c0 - 0.15, 0.6 * c0 + 0.1),
                       title="plan view (y across, x forward)", equal=True)
            xs = np.concatenate([etas * b2, (etas * b2)[::-1]])
            ys = np.concatenate([0.5 * cs, -0.5 * cs[::-1]])
            plot.fill(xs, ys, (70, 45, 20))
            plot.line(xs, ys, C_PANEL_ON, 2, closed=True)
            if spec.plate_h > 0:
                for sgn in (-1, 1):
                    plot.line([sgn * b2, sgn * b2], [-0.65 * c0 * lam, 0.65 * c0 * lam], C_PLATE, 3)
        # -- section
        r2 = ui.panel(screen, R(856, 12, 412, 226))
        coords = None
        try:
            coords = self.lib.airfoils[spec.airfoil].coords()
        except KeyError:
            pass
        if coords is not None:
            plot.begin(screen, r2, (-0.05, 1.05), (-0.3, 0.3), title=f"section {spec.airfoil}", equal=True)
            plot.fill(coords[:, 0], coords[:, 1], (60, 62, 70))
            plot.line(coords[:, 0], coords[:, 1], C_TEXT, 2, closed=True)
            plot.hline(0.0, ui.C_GRID)
            if self.polar is not None:
                p = self.polar
                g = self.lib.airfoils[spec.airfoil].geometry()
                plot.label(0.02, 0.26, f"t/c {100 * g['tc']:.1f}%  camber {100 * g['camber']:.1f}%", C_TEXT_DIM, 11)
                plot.label(0.02, 0.19, f"a {p.a_lin:.2f}/rad  alpha_L0 {p.alpha_L0_deg:+.1f} deg", C_TEXT_DIM, 11)
                plot.label(0.02, -0.22, f"{'XFOIL' if p.source == 'xfoil' else 'ESTIMATE'} polar  Re {p.re:.2g}", C_OK if p.source == "xfoil" else C_WARN, 11)
                plot.label(0.02, -0.28, f"cl_max {p.cl_max:.2f}  cd_min {1e4 * p.cd_min:.0f} ct", C_OK if p.source == "xfoil" else C_WARN, 11)
        # -- spanwise loading
        r3 = ui.panel(screen, R(444, 246, 404, 226))
        sd = self.span_data
        if sd is not None and self.polar is not None:
            y = np.asarray(sd["y"])
            cl = np.asarray(sd["cl"])
            eta = y / max(0.5 * spec.span, 1e-6)
            ymax = max(1.2, float(np.max(np.abs(cl))) * 1.15, self.polar.cl_max * 1.05)
            plot.begin(screen, r3, (-1.0, 1.0), (min(0.0, float(cl.min()) * 1.2), ymax),
                       title="strip cl across the span (at the mount incidence)", xlabel="y / (b/2)")
            plot.hline(self.polar.cl_max, C_WARN)
            plot.label(0.30, self.polar.cl_max, "section cl_max", C_WARN, 10, dy=3)
            plot.line(eta, cl, C_PANEL_ON, 2)
            aeff = np.asarray(sd["aeff"])
            plot.line(eta, aeff / 20.0, ui.C_LINE4, 1)
            plot.label(0.30, ymax * 0.97, f"CL {sd['CL']:.3f}  e {sd['e']:.3f}", C_TEXT, 11, dy=2)
            plot.label(-0.98, ymax * 0.97, "blue: alpha_eff / 20", ui.C_LINE4, 10, dy=2)
        # -- polar
        r4 = ui.panel(screen, R(856, 246, 412, 226))
        if self.polar is not None:
            p = self.polar
            plot.begin(screen, r4, (p.alpha.min(), p.alpha.max()), (min(p.cl.min(), -0.2) - 0.1, p.cl_max + 0.3),
                       title="section polar cl(alpha deg)   green: cd x 20")
            plot.line(p.alpha, p.cl, C_PANEL_ON, 2)
            plot.line(p.alpha, p.cd * 20.0, ui.C_LINE2, 1)
            plot.vline(p.alpha_valid[0], C_WARN)
            plot.vline(p.alpha_valid[1], C_WARN)
            if sd is not None:
                aeff = np.asarray(sd["aeff"])
                plot.points(aeff, p.cl_at(aeff), ui.C_LINE4, 2)
        # -- read-outs
        r5 = ui.panel(screen, R(444, 480, 824, 196))
        a = spec.aero
        x, y = r5.x + 12, r5.y + 8
        est = a.get("polar_is_estimate", True)
        text.blit(screen, f"{spec.name}   {SLOT_LABEL[self.key]}   "
                  f"S {spec.S:.3f} m2  AR {spec.AR:.2f}  MAC {spec.mac:.3f} m  Re {a.get('Re', 0):.2g}"
                  + ("   [ESTIMATE polar]" if est else "   [XFOIL polar]"), x, y, 13,
                  C_WARN if est else C_OK)
        y += 20
        if self.err:
            text.blit(screen, f"lattice refused: {self.err}", x, y, 13, C_WARN)
            y += 20
        if "CLa" in a:
            text.blit(screen, f"lift law  CL = {a['CL0']:+.3f} + {a['CLa']:.3f} alpha   clamped "
                      f"[{a['CL_min']:+.2f}, {a['CL_max']:+.2f}]   stall at {a['alpha_stall_deg']:+.1f} deg (crit. section)   e {a['e']:.3f}",
                      x, y, 13, C_TEXT)
            y += 18
            text.blit(screen, f"drag law  CD = {a['cd0']:.4f} {a['cd1']:+.4f} CL {a['cd2']:+.4f} CL^2   "
                      f"(fit err {1e4 * a['cd_fit_err']:.0f} ct, struts {1e4 * a['cd_strut']:.0f} ct)", x, y, 13, C_TEXT_DIM)
            y += 20
        dp = self.dp
        if dp:
            V = dp["V"]
            line = (f"AT {V:.1f} m/s, inc {slot.inc_deg:+.1f} deg:   CL {dp['CL']:.3f}   "
                    f"{'F' if self.role == 'flank' else 'Fz'} {dp['F']:.0f} N ({100 * dp['F'] / (CAR.m * G):.1f}% mg)   "
                    f"D {dp['D']:.1f} N   L/D {dp['LD']:.2f}   margin {dp['stall_margin_deg']:.1f} deg")
            text.blit(screen, line, x, y, 14, C_WARN if dp.get("stalled") else C_TEXT)
            y += 20
            if self.role == "flank":
                g = dp.get("gain_pct")
                gtxt = "runaway" if g is None else f"{g:+.2f}%"
                text.blit(screen, f"corner-speed gain at R = 100 m: {gtxt}   (x + b)/b x{(slot.x + CAR.b) / CAR.b:.2f}   "
                          f"cap +{GAIN_CAP_PCT:.2f}%", x, y, 13, C_OK if (g or 0) > 0 else C_TEXT_DIM)
            else:
                share = (slot.x + CAR.b) / CAR.L
                text.blit(screen, f"downforce split front {100 * share:.0f}% / rear {100 * (1 - share):.0f}%   "
                          f"ground effect at h {slot.h:.2f} m   deploys: {slot.mode}", x, y, 13,
                          C_OK if share > 0.3 else C_WARN)
            y += 20
        if self.lap is not None and self.lap.ok:
            d = self.lap.time - self.base_lap.time
            text.blit(screen, f"THE MISSION: {self.g.mission.track} ({self.g.mission.surface})   "
                              f"lap {self.lap.time:.4f} s   {d:+.4f} s vs {self.base_lap.time:.4f} s "
                              f"with this slot empty", x, y, 14,
                      ui.C_OK if d < 0.0 else C_WARN, bold=True)
            y += 20
        if self.result:
            rr = self.result
            fmt = ((lambda v: f"{-v:.4f} s") if rr["objective"] == "lap time"
                   else (lambda v: f"{v:.4g}"))
            text.blit(screen, f"optimiser: {rr['objective']}  BO {fmt(rr['bo'])}  random {fmt(rr['rs'])}  "
                      f"start {fmt(rr['start'])}   {rr['n']} evals {rr['secs']:.1f} s", x, y, 13, ui.C_SECTION)
            y += 18
            if rr.get("design"):                  # the winner, named in the page's row order
                text.blit(screen, rr["design"], x, y, 12, ui.C_DIM)
                y += 16
            tr = np.asarray([v if math.isfinite(v) else np.nan for v in rr["trace"]])
            rt = np.asarray([v if math.isfinite(v) else np.nan for v in rr["rs_trace"]])
            rr_rect = pygame.Rect(r5.right - 300, r5.y + 8, 288, 120)
            fin = np.concatenate([tr[np.isfinite(tr)], rt[np.isfinite(rt)]])
            if fin.size:
                lo, hi = float(fin.min()), float(fin.max())
                plot.begin(screen, rr_rect, (1, len(tr)), (lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9)),
                           title="best so far: BO (orange) vs random (grey)", grid=False)
                k = np.arange(1, len(tr) + 1)
                m = np.isfinite(rt)
                plot.line(k[m], rt[m], C_TEXT_DIM, 1)
                m = np.isfinite(tr)
                plot.line(k[m], tr[m], C_PANEL_ON, 2)
        if self.msg:
            text.blit(screen, self.msg[:120], x, r5.bottom - 20, 12, ui.C_KEY)
        return help_


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
            P("w", "RANKING WEIGHTS (AeroBO screen)", None, kind="label"),
            P("ld_cr", "L/D at design cl", lambda: self.weights["ld_cr"], self._w("ld_cr"), step=0.05, lo=0, hi=1),
            P("cl_max", "max lift (cl_max)", lambda: self.weights["cl_max"], self._w("cl_max"), step=0.05, lo=0, hi=1),
            P("ld_max", "efficiency (L/D max)", lambda: self.weights["ld_max"], self._w("ld_max"), step=0.05, lo=0, hi=1),
            P("thin", "thickness (thin)", lambda: self.weights["thin"], self._w("thin"), step=0.05, lo=0, hi=1),
            P("cm", "pitching moment (|cm|)", lambda: self.weights["cm"], self._w("cm"), step=0.05, lo=0, hi=1),
            P("cl", "design cl", lambda: self.cl_design, self._set_cl, step=0.1, fine=0.02, lo=0.0, hi=2.0),
            P("sort", "sort by", lambda: self.sort, self._set_sort, kind="choice",
              choices=["score", "cl_max", "ld_cr", "name", "tc"]),
            P("a", "ACTIONS", None, kind="label"),
            P("use", "use this section  (ENTER)", None, lambda _: self.g.assign_airfoil(), kind="action"),
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
            self.g.hint = f"XFOIL queued for {name}"

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
        self.focus = "wings"
        self.refresh()

    def refresh(self) -> None:
        keep_w = self.wings.current()[0] if self.wings.current() else None
        keep_b = self.builds.current()[0] if self.builds.current() else None
        items = []
        for name in sorted(self.lib.wings):
            w = self.lib.wings[name]
            a = w.aero
            if w.legacy:
                sub = f"published panel: CL0 {w.legacy['CL0']:.2f}  L/D {w.legacy['LD']:.1f}  S {w.legacy.get('S', 0.35):.2f}"
            elif "CLa" in a:
                sub = (f"{w.airfoil}  b {w.span:.2f} c {w.chord:.2f} taper {w.taper:.2f} plates {w.plate_h:.2f}  "
                       f"S {w.S:.2f}  CLa {a['CLa']:.2f}  CLmax {a['CL_max']:.2f}  {'est' if a.get('polar_is_estimate', True) else 'XFOIL'}")
            else:
                sub = f"{w.airfoil}  b {w.span:.2f} c {w.chord:.2f}  (not analysed)"
            items.append((name, sub, f"{w.role}{' *' if w.builtin else ''}"))
        self.wings.set_items(items, keep=keep_w)
        items = []
        for name in sorted(self.lib.builds):
            b = CarBuild.from_json(self.lib.builds[name])
            items.append((name, b.summary(self.lib)[:90], "build"))
        self.builds.set_items(items, keep=keep_b)

    def draw(self, screen, text: ui.Text, u: float) -> None:
        R = lambda x, y, w, h: (int(x * u), int(y * u), int(w * u), int(h * u))   # noqa: E731
        self.wings.draw(screen, text, ui.panel(screen, R(12, 12, 640, 336)), row_h=int(33 * u), size=13,
                        focus=(self.focus == "wings"))
        self.builds.draw(screen, text, ui.panel(screen, R(12, 356, 640, 320)), row_h=int(33 * u), size=13,
                         focus=(self.focus == "builds"), empty="(no saved builds: S saves the current car)")
        r = ui.panel(screen, R(660, 12, 608, 664))
        x, y = r.x + 12, r.y + 10
        b = self.g.build
        text.blit(screen, f"CURRENT CAR  '{b.name}'   selected slot: {SLOT_LABEL[self.g.sel]}", x, y, 14, ui.C_SECTION, bold=True)
        y += 24
        for key in SLOTS:
            s = b.slot(key)
            w = self.lib.wings.get(s.wing)
            text.blit(screen, f"{SLOT_LABEL[key]:12s} {w.name if w else 'none':18s} x {s.x:+.2f}  h {s.h:.2f}  inc {s.inc_deg:+.1f}"
                      + (f"  {s.mode}" if key == "top" else ""), x, y, 13, C_TEXT if w else C_TEXT_DIM)
            y += 20
        y += 10
        it = self.wings.current() if self.focus == "wings" else self.builds.current()
        if it and self.focus == "wings":
            w = self.lib.wings[it[0]]
            text.blit(screen, f"WING '{w.name}'  ({w.role}{', built-in' if w.builtin else ''})", x, y, 14, ui.C_SECTION, bold=True)
            y += 22
            lines = [w.notes or "", f"section {w.airfoil}   span {w.span:.3f} m   chord {w.chord:.3f} m   taper {w.taper:.2f}",
                     f"twist {w.twist_deg:+.1f} deg   plates {w.plate_h:.3f} m   S {w.S:.3f} m2   AR {w.AR:.2f}   MAC {w.mac:.3f} m"]
            a = w.aero
            if "CLa" in a:
                lines.append(f"CL = {a['CL0']:+.3f} + {a['CLa']:.3f} alpha  in [{a['CL_min']:+.2f}, {a['CL_max']:+.2f}]   e {a['e']:.3f}   "
                             f"polar {'ESTIMATE' if a.get('polar_is_estimate', True) else 'XFOIL'} Re {a.get('Re', 0):.2g}")
                lines.append(f"CD = {a['cd0']:.4f} {a['cd1']:+.4f} CL {a['cd2']:+.4f} CL^2")
                dp = design_point(w, 0.0)
                if dp:
                    lines.append(f"at {dp['V']:.1f} m/s, 0 deg:  CL {dp['CL']:.2f}  F {dp['F']:.0f} N  D {dp['D']:.1f} N  L/D {dp['LD']:.1f}")
            elif w.legacy:
                lines.append(f"closed form: CL0 {w.legacy['CL0']:.2f}, L/D {w.legacy['LD']:.1f}, S {w.legacy.get('S', 0.35):.2f} m2 (CONTRACT s4)")
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
            bb = CarBuild.from_json(self.lib.builds[it[0]])
            text.blit(screen, f"BUILD '{bb.name}'", x, y, 14, ui.C_SECTION, bold=True)
            y += 22
            for key in SLOTS:
                s = bb.slot(key)
                text.blit(screen, f"{SLOT_LABEL[key]:12s} {s.wing or 'none':18s} x {s.x:+.2f}  h {s.h:.2f}  inc {s.inc_deg:+.1f}",
                          x, y, 13, C_TEXT_DIM)
                y += 20
            y += 8
            text.blit(screen, "ENTER loads it as the current car", x, y, 12, C_OK)
        y = r.bottom - 60
        text.blit(screen, "TAB wings/builds   ENTER use/load   S save car as build   N new wing (designer)   "
                  "DEL delete (user items)   ESC back", x, y, 12, C_TEXT_DIM)
        if self.g.hint:
            text.blit(screen, self.g.hint[:90], x, y + 20, 12, ui.C_KEY)


# =========================================================================== #
#  THE EDITOR LOOP                                                             #
# =========================================================================== #
# =========================================================================== #
#  STEP 1 -- THE MISSION PAGE                                                 #
# =========================================================================== #
class MissionPage:
    """What the wing is for, stated before anything is designed.

    First of the two pages the garage now walks, and it is first for
    AeroBO's reason (`aero/mission.py` carries the measurement): a section and
    a planform cannot be judged until someone says what the wing is for, and
    "maximise a coefficient against a drag allowance" is a calibration rather
    than a requirement. Here the requirement is a LAP of one of carsim's own
    circuits, and the exchange rate between downforce and drag is the
    circuit's integral rather than a number the designer types.

    The page is GATED: nothing downstream opens until `state()` has been
    pressed. That is deliberate -- a mission nobody confirmed is a default.
    """

    def __init__(self, g: "Garage"):
        self.g = g
        self.result = None          # the car as it stands
        self.bare = None            # the same car with nothing on it
        self.err = ""
        P = ui.Param
        self.params = ui.ParamList([
            P("m", "MISSION  (what the wing is for)", None, kind="label"),
            P("track", "circuit", lambda: self.g.mission.track, self._set_track, kind="choice",
              choices=list(ms.TRACKS),
              help="carsim's own geometry: the lap is integrated over the arcs and "
                   "straights drive/track.py defines the circuit with. The dragstrip "
                   "is not offered -- with no corner, every wing on it is pure drag"),
            P("surf", "surface", lambda: self.g.mission.surface, self._set_surface, kind="choice",
              choices=[n for n, _ in ms.MissionSpec.SURFACES],
              help="tyre grip scale. 'wet' is published (qss.sweep, 0.55/0.87); "
                   "'damp' is an estimate and track.py says so"),
            P("r", "THE CAR AS IT STANDS", None, kind="label"),
            P("lap", "lap", lambda: (self.result.time if self.result else 0.0), None,
              lo=None, hi=None, unit="s", fmt="{:.3f}", enabled=False,
              help="quasi-steady: corners at their steady speed, straights the "
                   "acceleration profile met by the braking profile. An OPTIMUM, "
                   "not a prediction -- a driven lap is slower"),
            P("dlap", "vs the car with no wings", lambda: self._delta(), None,
              lo=None, hi=None, unit="s", fmt="{:+.3f}", enabled=False,
              help="what the wings currently fitted are worth. Negative is faster"),
            P("vm", "mean speed", lambda: (self.result.v_mean if self.result else 0.0), None,
              lo=None, hi=None, unit="m/s", fmt="{:.1f}", enabled=False,
              help="length / time. The section's Reynolds number is read at it"),
            P("vx", "fastest point", lambda: (self.result.v_max if self.result else 0.0), None,
              lo=None, hi=None, unit="m/s", fmt="{:.1f}", enabled=False),
            P("tc", "in corners", lambda: (self.result.t_corner if self.result else 0.0), None,
              lo=None, hi=None, unit="s", fmt="{:.2f}", enabled=False,
              help="how much of the lap the wing's downforce or side force is paid for by"),
            P("ts", "on straights", lambda: (self.result.t_straight if self.result else 0.0), None,
              lo=None, hi=None, unit="s", fmt="{:.2f}", enabled=False,
              help="how much of it the wing's drag is charged over. These two ARE the "
                   "exchange rate nobody has to state"),
            P("a", "ACTIONS", None, kind="label"),
            P("go", "state this mission  ->  design the section  (ENTER)", None,
              lambda _: self.state(), kind="action"),
        ], title="STEP 1 of 2   MISSION")
        self.update()

    # -- rows ----------------------------------------------------------------
    def _set_track(self, v):
        self.g.mission.track = str(v) if str(v) in ms.TRACKS else "arena"
        self.g.mission.stated = False
        self.update()

    def _set_surface(self, v):
        for nm, sc in ms.MissionSpec.SURFACES:
            if nm == str(v):
                self.g.mission.mu_scale = sc
        self.g.mission.stated = False
        self.update()

    def _delta(self) -> float:
        if self.result is None or self.bare is None:
            return 0.0
        return self.result.time - self.bare.time

    # -- analysis ------------------------------------------------------------
    def profile(self):
        return self.g.mission.profile(make_track)

    def update(self) -> None:
        self.err = ""
        try:
            pf = self.profile()
            mu = self.g.mission.mu_scale
            self.result = ms.lap(pf, self.g.build.mission_aero(self.lib), mu_scale=mu)
            self.bare = ms.lap(pf, ms.MissionAero(), mu_scale=mu)
        except Exception as exc:                      # a bad circuit never kills the page
            self.result = self.bare = None
            self.err = str(exc)

    @property
    def lib(self):
        return self.g.lib

    def state(self) -> None:
        if self.result is None or not self.result.ok:
            self.g.hint = f"this mission does not fly: {self.err or 'the lap did not close'}"
            return
        self.g.mission.stated = True
        self.g.open_section()

    # -- drawing --------------------------------------------------------------
    def draw(self, screen, text: ui.Text, plot: ui.Plot, u: float) -> str:
        R = lambda x, y, w, h: (int(x * u), int(y * u), int(w * u), int(h * u))   # noqa: E731
        help_ = self.params.draw(screen, text, ui.panel(screen, R(12, 12, 460, 664)),
                                 row_h=int(21 * u), size=13)
        pf = self.profile()

        # -- the circuit, drawn from its own centreline
        r1 = ui.panel(screen, R(484, 12, 400, 330))
        try:
            tr = make_track(self.g.mission.track)
            xy = tr.xy
            x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
            y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
            plot.begin(screen, r1, (x0, x1), (y0, y1), title=pf.title, equal=True)
            plot.line(xy[:, 0], xy[:, 1], col=ui.C_DIM, width=2, closed=tr.closed)
        except Exception:
            plot.begin(screen, r1, (0, 1), (0, 1), title=pf.title)

        # -- corner by corner, with which limit set the speed
        r2 = ui.panel(screen, R(484, 352, 400, 324))
        x, y = r2.x + 12, r2.y + 10
        text.blit(screen, pf.describe(), x, y, 13, ui.C_SECTION, bold=True)
        y += 22
        if self.result is not None and self.result.ok:
            text.blit(screen, "corner      R       V        limit", x, y, 12, ui.C_DIM)
            y += 17
            cs = pf.corners
            for i, (c, v, w) in enumerate(zip(cs, self.result.v_corner, self.result.limited)):
                col = ui.C_WARN if w == "power" else ui.C_TEXT
                text.blit(screen, f"T{i + 1:<2d}     {c.radius:6.0f} m  {v:6.2f} m/s   {w}",
                          x, y, 12, col)
                y += 16
            y += 6
            text.blit(screen, f"lap {self.result.time:.3f} s   "
                              f"{self.result.t_corner:.2f} s in corners / "
                              f"{self.result.t_straight:.2f} s on straights",
                      x, y, 12, ui.C_KEY)
            y += 18
            text.blit(screen, "the split above IS the exchange rate between", x, y, 11, ui.C_DIM)
            y += 14
            text.blit(screen, "side force and drag. Nobody has to state it.", x, y, 11, ui.C_DIM)
        elif self.err:
            text.blit(screen, self.err[:60], x, y, 12, ui.C_WARN)

        # -- what is stated, and what it gates
        r3 = ui.panel(screen, R(896, 12, 372, 664))
        x, y = r3.x + 12, r3.y + 10
        st = self.g.mission.stated
        text.blit(screen, "STATED" if st else "NOT YET STATED", x, y, 14,
                  ui.C_OK if st else ui.C_WARN, bold=True)
        y += 26
        for ln in (f"circuit   {pf.title}",
                   f"surface   {self.g.mission.surface}",
                   f"length    {pf.length:.0f} m",
                   f"corners   {len(pf.corners)}"):
            text.blit(screen, ln, x, y, 13)
            y += 19
        y += 10
        for ln in ("The section and the wing are both scored",
                   "against THIS mission, so the two stages",
                   "are comparable. Change the circuit and",
                   "it has to be stated again.",
                   "",
                   "Braking uses tyre grip only: brakes are",
                   "not modelled separately.",
                   "A wing is compared against a wing under",
                   "one assumption; the absolute time is not",
                   "a claim about the real car."):
            text.blit(screen, ln, x, y, 11, ui.C_DIM)
            y += 15
        return help_


# =========================================================================== #
#  STEP 2 -- THE DESIGN PAGE: a navigator over the whole procedure            #
# =========================================================================== #
#: The procedure, written down, in the order the UROP app's own navigator
#: carries it. Each group is a thing being designed and each step is a stage
#: of designing it; the two SECTION groups run the same four stages because
#: they are the same problem pointed at two different surfaces.
#:
#: ENDPLATE is a real group and not a courtesy: `vlm.Lattice` has always taken
#: `plate_a` / `plate_L0` and `wing.build_lattice` has always hardcoded them to
#: a flat plate, so giving the tip panels a designed section is a hook that was
#: already there. MEASURED on `flank-e423`: a cambered plate is worth +4.9 % of
#: CL0 through `library.analyse_wing`, and +21.6 % on the thicker-plated
#: reference wing `section.reference_problem` builds.
def _wrap(msg: str, n: int = 44) -> list:
    """A sentence broken to `n` characters, for the explanatory panels. The
    argument beside a weight IS the weight's documentation, so it is shown in
    full rather than clipped to one line."""
    out, line = [], ""
    for word in str(msg).split():
        if len(line) + len(word) + 1 > n:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


#: the criteria rows name the thing first, its symbol after (scr.META
#: keeps the symbol-first names for the bar chart and the docs)
PLAIN_CRIT = {"clmax": "max lift (cl_max)", "cm": "pitching moment (|cm|)",
              "ldmax": "efficiency (L/D max)"}


DESIGN_TREE = (
    ("AIRFOIL   (the wing's own section)",
     (("af.screen", "library screening"), ("af.rank", "ranking"),
      ("af.section", "section"), ("af.opt", "shape optimisation"))),
    ("ENDPLATE  (the tip panels' section)",
     (("ep.screen", "library screening"), ("ep.rank", "ranking"),
      ("ep.section", "section"), ("ep.opt", "shape optimisation"))),
    ("WING",
     (("w.type", "wing type"), ("w.box", "design box"),
      ("w.solver", "solver"), ("w.conv", "convergence"))),
    ("RESULTS",
     (("r.summary", "summary"), ("r.geometry", "geometry"),
      ("r.loading", "loading"), ("r.evals", "evaluations"))),
)


class SectionModel:
    """One section being designed, through the four stages the navigator
    lists: SCREEN the library, read the RANKING, edit the SECTION's own
    weights, then OPTIMISE the shape.

    `target` is 'main' (the wing's section) or 'plate' (the end plates'). The
    two are the same problem -- `aero/section.py` carries the difference -- so
    they are the same object here and the navigator shows them as the same
    four steps.
    """

    def __init__(self, page: "DesignPage", target: str):
        self.page, self.target = page, target
        self.prob: sec.SectionProblem | None = None
        self.x = None
        self.res: dict = {}
        self.run: dict | None = None
        self.ranked: list = []          # (name, composite, res, x), best first
        self.clipped: list = []         # library sections the box had to clip
        self.refused: list = []         # (name, why) -- the gates' own work
        self.seed = ""
        self.objective = list(sec.SECTION_OBJECTIVES)[0]
        #  the budget is AeroBO's measured LAW at this problem's own row
        #  count, not a hand-picked number: `optimize.budget_for`. 48 was a
        #  guess in the one place that repository has a measurement.
        self.effort = "balanced"
        self.budget = opt.budget_for(2 * sec.N_CST + (1 if target == "plate" else 2),
                                     self.effort)
        self.msg = ""
        #  THE WEIGHTS ARE THE SCREEN'S QUESTION, not the search's. They open
        #  on this target's recommended preset -- AeroBO's own, `screen.py`
        #  names which -- and editing one makes the set the user's, after
        #  which nothing moves it again.
        self.preset = scr.recommended(SLOT_ROLE.get(page.key, "flank"), target)
        self.weights = dict(scr.PRESETS[self.preset])
        self.gates = dict(scr.GATES_OFF_DEFAULT)
        self.floors = dict(scr.FLOORS_OFF)
        #: the lift this surface is SCREENED at. A plain reference, not a
        #: derived requirement -- see `screen.REFERENCE_CL` -- and editable,
        #: because it is the one number the whole ranking turns on.
        self.cl_design = 0.0 if target == "plate" else scr.REFERENCE_CL
        self.band: dict | None = None
        self.seed_sub: dict | None = None
        #: has a section FROM THIS GROUP been put on the wing? That is what
        #: finishes the group -- not the optimiser, which AeroBO is explicit
        #: is optional ("Step 2 is optional -- that is the point of ordering
        #: them this way"). Screen, take the winner, fit it: the group is
        #: done in three keys without a search.
        self.fitted = False
        #: ...or, for the plates, the decision NOT to give them one. An
        #: explicit answer, because "I looked and chose flat" and "I never
        #: came here" are different states and only one of them finishes a
        #: step.
        self.declined = False
        self.params = ui.ParamList(self._weight_rows(), title="")
        self.opt_params = ui.ParamList(self._opt_rows(), title="")
        self.screen_params = ui.ParamList(self._screen_rows(), title="")

    # -- the criterion weights, the gates and the floors ---------------------
    def _set_w(self, crit: str):
        def f(v):
            self.weights[crit] = max(0.0, float(v))
            self.preset = "(edited)"
            self._push()
        return f

    def _set_gate(self, key: str):
        def f(v):
            self.gates[key] = float(v)
            self._push()
        return f

    def _set_floor(self, key: str):
        def f(v):
            self.floors[key] = float(v)
            self._push()
        return f

    def _set_preset(self, name: str):
        self.preset = str(name)
        if self.preset in scr.PRESETS:
            self.weights = dict(scr.PRESETS[self.preset])
        self._push()

    def _push(self) -> None:
        """Hand the form's state to the problem. The screen and the three
        composite objectives read the SAME three dicts, which is the whole
        point: the shortlist is chosen on the map it is judged on."""
        if self.prob is not None:
            self.prob.weights = self.weights
            self.prob.gates = self.gates
            self.prob.floors = self.floors

    def dead(self) -> dict:
        """Criteria that cannot rank THIS target, and why."""
        return scr.DEAD.get(self.target, {})

    def dead_weighted(self) -> list:
        """...and the ones the user has nonetheless put weight on."""
        return [k for k in self.dead() if float(self.weights.get(k, 0.0)) > 0.0]

    def _screen_rows(self):
        P = ui.Param
        dead = self.dead()
        rows = [
            P("s", "WHAT THIS SECTION IS FOR  (the screen's weights)", None, kind="label"),
            P("preset", "preset", lambda: self.preset, self._set_preset,
              kind="choice", choices=list(scr.PRESETS) + ["(edited)"],
              help="AeroBO's own shipped sets. 'wing' is its bulk-sweep preset, "
                   "which is what it hands a surface that carries the design load; "
                   "'plate' is the entry its car endplate shares with its fin. "
                   "Moving any weight below makes the set yours"),
        ]
        for k in scr.CRITERIA:
            label, why = scr.META[k]
            label = PLAIN_CRIT.get(k, label)
            if k in dead:
                label += "  (ranks nothing)"          # short: the value sits on the same row
            rows.append(P(f"w.{k}", label, (lambda kk=k: float(self.weights.get(kk, 0.0))),
                          self._set_w(k), step=0.05, fine=0.01, lo=0.0, hi=1.0,
                          fmt="{:.2f}",
                          help=(dead[k] if k in dead else why)))
        rows += [
            P("n", "normalised", lambda: "sum to 1 for the score", None,
              kind="choice", choices=[], enabled=False,
              help="the weights are divided by their own sum, so they read as "
                   "fractions of one score whatever numbers are typed"),
            P("cl", "screened at cl",
              lambda: (0.0 if self.target == "plate" else self.cl_design),
              (None if self.target == "plate" else self._set("cl_design")),
              step=0.05, fine=0.01, lo=0.0, hi=3.0, fmt="{:.3f}",
              enabled=(self.target != "plate"),
              help="a plain REFERENCE, not a derived requirement: these wings are not "
                   "sized to carry a stated load, they are asked for as much downforce "
                   "as the lap will pay for, so there is no design lift to derive. "
                   "AeroBO's own answer to the same question (REFERENCE_CL = 1.0). "
                   "An END PLATE is fixed at 0 -- its panels are vertical and carry no "
                   "design load, which is why three criteria above are dead on it"),
            P("g", "GATES  (a candidate is dropped, not ranked low)", None, kind="label"),
            P("tcmin", "minimum t/c", lambda: float(self.gates.get("tc_min", 0.0)),
              self._set_gate("tc_min"), step=0.005, fine=0.001, lo=0.0, hi=0.20,
              fmt="{:.4f}",
              help="0 = off, which is the default. AeroBO gates at 0.15 over an "
                   "AIRCRAFT library; carsim's 34 race sections run 0.043 to 0.161 "
                   "with a median of 0.107, so 0.15 would admit one of them. "
                   "0.0956 is this library's own lower quartile"),
            P("cmmax", "maximum |cm|", lambda: min(float(self.gates.get("cm_max", 1e9)), 1.0),
              self._set_gate("cm_max"), step=0.02, fine=0.005, lo=0.0, hi=1.0,
              fmt="{:.4f}",
              help="1.0 = off. AeroBO gates at 0.08 because a tail trims the wing's "
                   "moment; a wing bolted to a car has no tail and the MOUNT carries "
                   "it, and the sections a downforce wing exists for are exactly the "
                   "ones over 0.08 -- S1223 0.348, S1210 0.306, CH10 0.275, E423 0.247"),
            P("f", "FLOORS  (off at the bottom of their range)", None, kind="label"),
            P("fclmax", "minimum cl_max", lambda: float(self.floors.get("clmax", 0.0)),
              self._set_floor("clmax"), step=0.05, fine=0.01, lo=0.0, hi=2.5, fmt="{:.2f}"),
            P("fldcr", "minimum L/D at the cl", lambda: float(self.floors.get("ldcr", 0.0)),
              self._set_floor("ldcr"), step=1.0, fine=0.25, lo=0.0, hi=150.0, fmt="{:.1f}"),
            P("fastall", "minimum stall angle", lambda: float(self.floors.get("astall", -90.0)),
              self._set_floor("astall"), step=0.5, fine=0.1, lo=-90.0, hi=25.0,
              unit="deg", fmt="{:+.1f}"),
            P("r", "SCREEN", None, kind="label"),
            P("read", "or read it off the wing as it stands", None,
              lambda _: self.read_wing_cl(), kind="action",
              enabled=(self.target != "plate"),
              help="the area-weighted mean of the lattice's own strip lift "
                   "coefficients. Offered, never taken automatically: it moves "
                   "whenever the wing's incidence moves, and a shortlist that "
                   "changes under another page's row is not a shortlist"),
            P("go", "screen the whole library  (L)", None,
              lambda _: self.page.screen(), kind="action",
              help="scores every section in the library on the weights above and "
                   "opens the RANKING with the result"),
            P("nn", "sections ranked", lambda: float(len(self.ranked)), None,
              lo=None, hi=None, kind="int", enabled=False),
            P("nr", "refused by a gate", lambda: float(len(self.refused)), None,
              lo=None, hi=None, kind="int", enabled=False,
              help="a gate DROPS a candidate before the band is measured, so the "
                   "0-100 sub-scores span only the sections that survived it"),
            P("clip", "clipped into the box", lambda: float(len(self.clipped)), None,
              lo=None, hi=None, kind="int", enabled=False,
              help="a library section outside the design box is judged at the CLIPPED "
                   "shape, which is not the section it came from. Counted, not hidden"),
            P("best", "best", lambda: (self.ranked[0][0] if self.ranked else "-"), None,
              kind="choice", choices=[], enabled=False),
        ]
        if self.target == "plate":
            #  THE WAY PAST THIS GROUP WITHOUT DESIGNING ANYTHING. The steps
            #  are gated in order, so a user who wants flat plates has to be
            #  able to SAY so -- otherwise the gate would make designing one
            #  compulsory, which is not a decision this page gets to take.
            rows.append(P("flat", "or fly FLAT plates and move on  (ENTER)", None,
                          lambda _: (self.decline(), self.page.advance("ep")), kind="action",
                          help="the lattice's own default, and what every wing in the "
                               "library was analysed with. An explicit answer: 'I "
                               "looked and chose flat' and 'I never came here' are "
                               "different states, and only one of them finishes a step"))
        return rows

    # -- the design vector as editable rows ---------------------------------
    def _set_x(self, i: int):
        def f(v):
            if self.x is None:
                return
            b = self.prob.bounds()
            self.x = np.asarray(self.x, dtype=float).copy()
            self.x[i] = min(max(float(v), b[i, 0]), b[i, 1])
            self.evaluate()
        return f

    def _get_x(self, i: int):
        return lambda: (float(self.x[i]) if self.x is not None and i < len(self.x) else 0.0)

    def _weight_rows(self):
        P = ui.Param
        rows = [P("w", "THE SHAPE  (the search's design vector, in order)", None,
                  kind="label")]
        for i in range(sec.N_CST):
            rows.append(P(f"wu{i}", f"w_up[{i}]", self._get_x(i), self._set_x(i),
                          step=0.01, fine=0.002, lo=-1.0, hi=1.0, fmt="{:+.4f}",
                          help="a CST (Kulfan) weight of the UPPER surface. The box is the "
                               "34 shipped sections' own per-coefficient hull, padded 15 %"))
        for i in range(sec.N_CST):
            j = sec.N_CST + i
            rows.append(P(f"wl{i}", f"w_lo[{i}]", self._get_x(j), self._set_x(j),
                          step=0.01, fine=0.002, lo=-1.0, hi=1.0, fmt="{:+.4f}",
                          help="a CST shape weight of the LOWER surface: each one moves a "
                               "stretch of it, nose to tail in order"))
        rows.append(P("tc", "t/c", self._get_x(2 * sec.N_CST), self._set_x(2 * sec.N_CST),
                      step=0.005, fine=0.001, lo=sec.TC_BOUNDS[0], hi=sec.TC_BOUNDS[1],
                      fmt="{:.4f}",
                      help="its own row, not whatever the weights give: the weights are "
                           "rescaled to land it exactly, and the rescale is linear so a "
                           "non-crossing section stays non-crossing"))
        if self.target != "plate":
            i = 2 * sec.N_CST + 1
            priced = (self.prob is None or self.prob.prices_inc)
            rows.append(P("inc", "incidence" + ("" if priced else "   (this objective does not price it)"),
                          self._get_x(i), self._set_x(i),
                          step=0.5, fine=0.1, lo=-8.0, hi=18.0, unit="deg", fmt="{:+.2f}",
                          help="over the SAME band the wing's own incidence row uses, and it "
                               "SEEDS that row -- the same question asked over two bands "
                               "would make the two stages incomparable. A COMPOSITE "
                               "objective never builds the lattice, so it cannot see this "
                               "row: it is left where it is and is NOT carried to the wing"))
        else:
            rows.append(P("incf", "judged at the wing's incidence",
                          lambda: (self.prob.inc_fixed if self.prob else 0.0), None,
                          lo=None, hi=None, unit="deg", fmt="{:+.2f}", enabled=False,
                          help="a plate has NO incidence row: the lattice's plates are "
                               "vertical panels carrying only a slope and a zero-lift angle. "
                               "This is the WING's angle, and it belongs to the wing page"))
        rows += [
            P("d", "WHAT IT IS", None, kind="label"),
            P("cam", "camber", lambda: float(self.res.get("geometry", {}).get("camber", 0.0)),
              None, lo=None, hi=None, fmt="{:+.4f}", enabled=False),
            P("clmax", "cl_max", lambda: float(self.res.get("cl_max", 0.0)), None,
              lo=None, hi=None, fmt="{:.3f}", enabled=False,
              help="ESTIMATE: a camber and thickness correlation, not a measured stall"),
            P("ldmax", "L/D max", lambda: float(self.res.get("ld_max", 0.0)), None,
              lo=None, hi=None, fmt="{:.1f}", enabled=False,
              help="ESTIMATE: friction + form factor + a lift-dependent term, +-15 %"),
            P("a0", "zero-lift angle", lambda: float(self.res.get("alpha_L0_deg", 0.0)), None,
              lo=None, hi=None, unit="deg", fmt="{:+.2f}", enabled=False,
              help="from the panel method: inviscid, and exact for what it models. For an "
                   "END PLATE this row is the whole effect -- the plate's slope barely enters"),
            P("dlap", "lap, vs the car without it",
              lambda: float(self.res.get("d_lap", 0.0)), None,
              lo=None, hi=None, unit="s", fmt="{:+.4f}", enabled=False),
            P("a", "ACTIONS", None, kind="label"),
            P("fit", "fit this section to the wing  (F)", None,
              lambda _: self.page.fit_and_advance(self), kind="action",
              help="saves it into the airfoil library and puts it on the wing. Until "
                   "this is pressed the wing is still flying whatever it was flying"),
        ]
        return rows

    def _opt_rows(self):
        P = ui.Param
        return [
            P("o", "SHAPE OPTIMISATION", None, kind="label"),
            P("obj", "objective", lambda: self.objective, self._set("objective"),
              kind="choice", choices=list(sec.SECTION_OBJECTIVES)),
            P("effort", "effort", lambda: self.effort, self._set_effort,
              kind="choice", choices=list(opt.EFFORTS),
              help="how much of the reachable improvement the budget is sized for: "
                   "quick 90 %, balanced 95 %, thorough 99 %. AeroBO's own three "
                   "settings, and its own measured law behind them"),
            P("budget", "evaluations", lambda: self.budget, self._set("budget"),
              kind="int", lo=8, hi=160,
              help="AeroBO's measured law, evals = a + b x d, fitted over 13 cases: "
                   "9.61 + 3.08 d at 95 %. The residual RMS is 10.7 evaluations, so it "
                   "is a sizing rule and not a prediction -- editable for that reason"),
            P("split", "Sobol start / BO",
              lambda: "%d + %d" % opt.split_for(self.dim(), self.budget), None,
              kind="choice", choices=[], enabled=False,
              help="0.5 x d initial points, clamped to [4, 16]. MEASURED: over 15 of "
                   "AeroBO's cases the seed sizes rank 0.5 -> 1.00, 1.0 -> 2.25, "
                   "2.0 -> 2.88, 4.0 -> 4.25 (mean rank, lower better). carsim used to "
                   "spend 16 of 48 here, which is 1.6 x d"),
            P("seed", "seeded from", lambda: (self.seed or "-"), None,
              kind="choice", choices=[], enabled=False,
              help="the section the search starts on, and -- for the two "
                   "seed-referenced objectives -- the floor it is held to. The "
                   "RANKING step's ENTER changes it"),
            P("gate", "held to the screen's gates",
              lambda: ("t/c >= %.4f" % self.gates["tc_min"]
                       if self.gates.get("tc_min", 0.0) > 0.0 else "no gates set"),
              None, kind="choice", choices=[], enabled=False,
              help="a search is refused the same candidates the screen was, so the "
                   "winner is comparable with the library it was chosen from"),
            P("run", "run the optimiser  (O)", None, lambda _: self.optimise(), kind="action"),
            P("r", "THE RUN", None, kind="label"),
            P("bo", "BO", lambda: self._score(self.run, "bo"), None,
              lo=None, hi=None, fmt="{:.4f}", enabled=False),
            P("rs", "random, same budget", lambda: self._score(self.run, "rs"), None,
              lo=None, hi=None, fmt="{:.4f}", enabled=False,
              help="the twin AeroBO's crossover map asks for: what the GP actually bought"),
            P("st", "start", lambda: self._score(self.run, "start"), None,
              lo=None, hi=None, fmt="{:.4f}", enabled=False),
        ]

    def _set(self, attr):
        def f(v):
            setattr(self, attr, v)
            if self.prob is not None:
                self.prob.objective = self.objective
            if attr == "objective":
                #  whether the incidence row is PRICED depends on the
                #  objective, and the row says so, so the form follows it
                keep = self.params.idx
                self.params = ui.ParamList(self._weight_rows(), title="")
                self.params.idx = min(keep, len(self.params.params) - 1)
                self.page.invalidate()
        return f

    def dim(self) -> int:
        """Rows in this problem's design vector -- what the budget law and
        the Sobol rule are both functions of."""
        return 2 * sec.N_CST + (1 if self.target == "plate" else 2)

    def _set_effort(self, v) -> None:
        self.effort = str(v) if str(v) in opt.EFFORTS else "balanced"
        self.budget = opt.budget_for(self.dim(), self.effort)

    def _score(self, run, key) -> float:
        if not run:
            return 0.0
        v = float(run.get(key, 0.0))
        return -v if (run.get("objective", "").startswith("lap") and math.isfinite(v)) else v

    def fmt_score(self, v: float) -> str:
        if not math.isfinite(v):
            return "refused"
        if self.objective.startswith("lap"):
            return f"{-v:.4f} s"
        #  a composite is 0-100 POINTS on the frozen band, and the two
        #  seed-referenced variants are read on the same scale -- so they are
        #  not printed to four decimals as if they were a coefficient
        return f"{v:.2f} pts" if scr.is_composite(self.objective) else f"{v:.4f}"

    # -- the problem ---------------------------------------------------------
    def rebuild(self) -> None:
        g = self.page.g
        key = self.page.key
        slot = g.build.slot(key)
        role = SLOT_ROLE[key]
        ride = slot.h if role == "top" else RIDE_H0["flank"]
        ref = self.page.wing.spec.copy()
        main_polar = None
        if self.target == "plate":
            #  the wing's own section, HELD while the plate moves
            main_polar = g.lib.wing_polar(ref)
            ref.plate_h = max(ref.plate_h, 0.06)
        self.prob = sec.SectionProblem(
            role=role, target=self.target, main_polar=main_polar, ref=ref,
            profile=g.mission.profile(make_track),
            base=g.build.mission_aero(g.lib, exclude=key),
            inc_bounds=BOUNDS[role]["inc_deg"], inc_fixed=slot.inc_deg,
            x=slot.x, h=slot.h, ride_h=ride, V_ref=V_REF[role],
            top_mode=slot.mode, objective=self.objective, mu_scale=g.mission.mu_scale,
            weights=self.weights, gates=self.gates, floors=self.floors,
            cl_design=self.design_cl(), band=self.band, seed_sub=self.seed_sub)

    def design_cl(self) -> float:
        """The lift THIS surface is screened at.

        A PLATE is screened at zero and cannot be moved off it: the lattice's
        tip panels are vertical and carry no design load, and that zero is
        exactly what retires three of the seven criteria (`screen.DEAD`).
        Everything else is screened at the reference the form holds.
        """
        return 0.0 if self.target == "plate" else float(self.cl_design)

    def wing_cl(self) -> float:
        """What the REFERENCE WING is actually asking its section for, right
        now -- the area-weighted mean of the lattice's own strip lift
        coefficients. Offered as an action, never taken automatically: it
        moves whenever the wing's incidence moves, and a shortlist that
        changes under another page's row is not a shortlist."""
        w = self.page.wing
        sd = (w.span_data if w is not None else None) or {}
        cl, c = np.asarray(sd.get("cl", [])), np.asarray(sd.get("c", []))
        if cl.size and c.size and c.sum() > 0:
            return float((cl * c).sum() / c.sum())
        return float((w.dp or {}).get("CL", scr.REFERENCE_CL)) if w else scr.REFERENCE_CL

    def read_wing_cl(self) -> None:
        self.cl_design = round(self.wing_cl(), 3)
        self.msg = (f"screening lift set to {self.cl_design:.3f} -- what the wing "
                    f"asks its section for at the angle it is bolted on at NOW. "
                    f"It moves when that angle moves; the ranking will not follow it")

    def seed_from(self, name: str) -> None:
        g = self.page.g
        try:
            coords = g.lib.airfoils[name].coords()
        except Exception:
            coords = af.naca4_coords("2412")
        self.seed = name
        self.x = self.prob.x0(coords, self.prob.inc_fixed)
        self.evaluate()

    def evaluate(self) -> None:
        if self.prob is None or self.x is None:
            return
        self.prob.objective = self.objective
        self.prob.band, self.prob.seed_sub = self.band, self.seed_sub
        self._push()
        self.res = sec.evaluate_section(self.x, self.prob)
        self.page.on_section_changed(self)

    # -- stage 1: screen the library ----------------------------------------
    def screen_library(self) -> None:
        """Score EVERY library section on the user's weights, and rank them.

        AeroBO's procedure, in AeroBO's two passes:

        1.  Build every library section inside the design box, read its seven
            criteria off its own polar, and apply the GATES and FLOORS. This
            pass never flies the lattice -- 3.5 ms a candidate.
        2.  Measure the frozen normalisation BAND over what survived, score
            everything on it, and rank. The band is kept, because it is what
            makes the composite a fixed function of the shape: a live min-max
            would move with the population and a search could not maximise it.
            *The shortlist is chosen on the map it is judged on* -- AeroBO's
            own rule, and the name of the test it broke.

        Pass 2 re-evaluates at the PAGE's objective, so a screen run with the
        lap selected also carries every candidate's lap delta. That costs the
        lattice (about 30 ms a candidate); a composite objective does not.

        A section whose CST shape weights or thickness fall outside the design box
        is judged at the CLIPPED shape, which is not the section it came from.
        Those are counted and named rather than quietly ranked.
        """
        g = self.page.g
        if self.prob is None:
            self.rebuild()
        self._push()
        self.prob.objective = self.objective
        self.prob.cl_design = self.design_cl()
        b = self.prob.bounds()

        #  pass 1: the shape, the polar, the seven criteria, the gates
        keep_band, keep_obj = self.prob.band, self.prob.objective
        self.prob.band, self.prob.objective = None, "composite"
        built, clipped, refused = [], [], []
        for name in sorted(g.lib.airfoils):
            try:
                coords = g.lib.airfoils[name].coords()
            except Exception:                                    # noqa: BLE001
                continue
            raw = np.concatenate(
                [np.concatenate(af.fit_cst(coords, sec.N_CST)),
                 [af.geometry(coords)["tc"]]]
                + ([[self.prob.inc_fixed]] if self.prob.has_inc else []))
            x = np.clip(raw, b[:, 0], b[:, 1])
            if float(np.max(np.abs(x - raw))) > 1e-9:
                clipped.append(name)
            r = sec.evaluate_section(x, self.prob)
            m, why = r.get("metrics"), r.get("gate")
            if m is None:
                refused.append((name, r.get("refused") or "no polar"))
            elif why:
                refused.append((name, why))
            else:
                built.append((name, x, m))
        self.prob.objective = keep_obj

        if not built:
            self.prob.band = keep_band
            self.ranked, self.clipped, self.refused = [], clipped, refused
            self.msg = (f"every one of the {len(refused)} library sections was "
                        f"refused before it could be ranked -- loosen the gates")
            return

        #  pass 2: the frozen band, then the score
        self.band = self.prob.band = scr.bands([m for _, _, m in built])
        rows = []
        for name, x, _m in built:
            r = sec.evaluate_section(x, self.prob)
            rows.append((name, float(r.get("composite", -math.inf)), r, x))
        rows.sort(key=lambda t: (-t[1] if math.isfinite(t[1]) else math.inf, t[0]))
        self.ranked, self.clipped, self.refused = rows, clipped, refused
        name, score, res, x = rows[0]
        self.seed, self.x, self.res = name, x, res
        #  the WINNER is the reference point the goal and Tchebycheff
        #  objectives are measured against, so the search starts from a design
        #  it must not go below rather than from nothing.
        self.seed_sub = self.prob.seed_sub = res.get("sub")
        self.page.on_section_changed(self)
        bad = self.dead_weighted()
        self.msg = (f"screened {len(built) + len(refused)} sections on your weights: "
                    f"'{name}' wins at {score:.2f} points"
                    + (f"   ({len(refused)} refused by a gate)" if refused else "")
                    + (f"   ({len(clipped)} clipped into the box)" if clipped else "")
                    + (f"   -- {', '.join(bad)} carries weight but ranks nothing here"
                       if bad else ""))

    def use_ranked(self, i: int) -> None:
        if 0 <= i < len(self.ranked):
            name, score, res, x = self.ranked[i]
            self.seed, self.x, self.res = name, np.asarray(x, dtype=float), res
            #  choosing a section also moves the reference point: the seed is
            #  what "none below the seed" is measured against, and a floor
            #  taken from a section the user did not pick would be a
            #  requirement nobody stated.
            self.seed_sub = res.get("sub")
            if self.prob is not None:
                self.prob.seed_sub = self.seed_sub
            self.page.on_section_changed(self)
            self.msg = (f"loaded '{name}' into the shape -- it is now the seed, "
                        f"and the floor the goal objective holds")

    # -- stage 4: optimise ---------------------------------------------------
    def optimise(self) -> dict | None:
        if self.prob is None:
            self.rebuild()
        self.prob.objective = self.objective
        self.prob.band, self.prob.seed_sub = self.band, self.seed_sub
        self._push()
        #  A COMPOSITE SEARCH CANNOT START BEFORE THE SCREEN HAS RUN, and the
        #  refusal names the step rather than reporting a failed run: the
        #  0-100 map is measured over the screened library, so without it
        #  there is nothing for the search to maximise. This is the
        #  navigator's order made binding, not a check bolted on beside it.
        if scr.is_composite(self.objective) and not self.band:
            self.msg = ("this objective is scored on a band measured over the "
                        "library -- screen it first (L), then optimise")
            return None
        if self.x is None:
            self.seed_from(self.seed or sorted(self.page.g.lib.airfoils)[0])
        bounds = self.prob.bounds()
        labels = self.prob.labels()
        x0 = np.clip(np.asarray(self.x, dtype=float), bounds[:, 0], bounds[:, 1])
        f = lambda x: sec.f_section(x, self.prob)                      # noqa: E731
        n = int(self.budget)
        #  AeroBO's measured split, not half the budget -- see
        #  `optimize.N_INIT`. The refusal rate of a uniform draw over this box
        #  is 36-42 %, well inside where the unconstrained rule applies.
        n_init, n_iter = opt.split_for(len(x0), n)
        t0 = time.perf_counter()
        bo = opt.maximise(f, bounds, n_init=n_init, n_iter=n_iter,
                          seed=0, x0=x0, labels=labels)
        rs = opt.random_search(f, bounds, n=n, seed=1, labels=labels)
        dt = time.perf_counter() - t0
        f0 = f(x0)
        self.run = dict(bo=bo["f_best"], rs=rs["f_best"], start=f0, trace=bo["best_trace"],
                        rs_trace=rs["best_trace"], n=n, secs=dt, objective=self.objective)
        if math.isfinite(bo["f_best"]) and bo["f_best"] >= f0:
            self.x = np.asarray(bo["x_best"], dtype=float)
            self.evaluate()
            self.msg = (f"BO {self.fmt_score(bo['f_best'])} vs random "
                        f"{self.fmt_score(rs['f_best'])} vs start {self.fmt_score(f0)}  "
                        f"({n} evals, {dt:.1f} s) - applied")
        elif not math.isfinite(bo["f_best"]):
            self.msg = f"every candidate was refused ({n} evals)"
        else:
            self.msg = (f"nothing beat the start ({n} evals, {dt:.1f} s): "
                        f"BO {self.fmt_score(bo['f_best'])} vs {self.fmt_score(f0)}")
        return self.run

    # -- what the navigator asks ---------------------------------------------
    def state(self, stage: str) -> str:
        """One of 'done' / 'ready' / 'blocked', for the navigator's mark and
        for whether the step may be selected at all.

        THE FOUR ARE A SEQUENCE, and the gate is real: a step is blocked
        until the step before it is done. That is what these four ARE -- the
        library is screened, the ranking it produced is read, the section it
        picked is shaped, and the search is seeded from that shape. Each gate
        opens with one action, so nothing is ever trapped.
        """
        if stage == "screen":
            return "done" if self.ranked else "ready"
        if stage == "rank":
            if not self.ranked:
                return "blocked"
            return "done" if self.x is not None else "ready"
        if stage == "section":
            if self.x is None or not self.ranked:
                return "blocked"
            return "done" if not self.res.get("refused") else "ready"
        #  the shape optimiser. Seeded from the section, and -- on a
        #  composite objective -- scored on the band the screen measured.
        if self.x is None or not self.ranked or self.res.get("refused"):
            return "blocked"
        if self.run:
            return "done"
        if scr.is_composite(self.objective) and not self.band:
            return "blocked"
        return "ready"

    def reason(self, stage: str) -> str:
        """Why a step is shut. A greyed-out step with no explanation is the
        thing this navigator exists to avoid -- AeroBO's `stage_states` keeps
        exactly this sentence beside every locked node."""
        what = "end plate" if self.target == "plate" else "section"
        if stage == "rank" and not self.ranked:
            return ("there is no ranking until the library has been screened "
                    "-- go to LIBRARY SCREENING and press L")
        if stage in ("section", "opt"):
            if not self.ranked:
                return (f"the {what} shaped here is the one the SCREEN picks. "
                        f"Screen the library first (L)")
            if self.x is None:
                return ("take a section from the RANKING first -- ENTER on "
                        "the one you want")
            if stage == "opt" and self.res.get("refused"):
                return (f"this {what} cannot be flown, so there is nothing to "
                        f"seed a search with: {self.res['refused'][:60]}")
            if stage == "opt" and scr.is_composite(self.objective) and not self.band:
                return ("this objective is scored on a band measured over the "
                        "library -- screen it first (L)")
        return ""

    def finished(self) -> bool:
        """Is this group DONE, as far as the group after it is concerned?

        Fitting the section to the wing is what finishes it. The search is
        NOT part of the answer: AeroBO's own docstring says step 2 is
        optional, and a screen whose winner you accepted is a complete
        decision.
        """
        return bool(self.fitted or self.declined)

    def decline(self) -> None:
        """Answer the plates' question with 'flat', and move on. An explicit
        answer, so a user who does not want a designed plate is not made to
        design one -- and so the navigator can still tell 'decided' from
        'never visited'."""
        self.declined = True
        self.msg = ("the end plates will fly FLAT -- the lattice's own "
                    "default, and what every wing in the library was "
                    "analysed with")


class DesignPage:
    """Everything downstream of the mission, behind one navigator.

    The left column is `DESIGN_TREE` -- the procedure written down, so what
    the garage is doing is legible before anything is pressed. TAB moves the
    focus between the navigator and the selected step's own rows; UP/DOWN
    drives whichever has it.

    This page replaces the separate SECTION and WING pages. They were two
    pages doing four stages each with no way to see the stages, which is the
    thing the navigator fixes.
    """

    #: WING TYPE -- the choices that decide WHAT is being built and searched,
    #: as opposed to the numbers inside it. `sized` changes the LENGTH of the
    #: design vector and `psec` changes the lattice, so neither is a box row.
    #: The plate's SECTION is not here: it is the ENDPLATE group's answer,
    #: and asking it twice is how the two come to disagree. The solver step
    #: shows what that group decided, as a read-out.
    TYPE_KEYS = ("mount", "blend", "bshape", "junc", "sized", "x", "h", "mode")
    #: DESIGN BOX -- the design vector as it stands, the two numbers derived
    #: from it, and the BUDGETS the objectives are held to. The bands
    #: themselves are appended from `Designer._box_rows`, because how many
    #: there are depends on whether the area is searched.
    BOX_KEYS = ("taper", "twistr", "twist", "inc", "plate", "ride", "span",
                "chord", "area", "cap", "floor")
    #: SOLVER -- the lattice the wing is solved on, and the SEARCH that runs
    #: it: objective, effort, budget, split, run. The convergence step is a
    #: READ-OUT of that run, with AeroBO's "keep going" beside it.
    SOLVER_KEYS = ("obj", "effort", "budget", "split", "run")
    CONV_KEYS = ("more", "go")

    def __init__(self, g: "Garage"):
        self.g = g
        self.key = "left"
        self.wing: Designer | None = None
        self.af = SectionModel(self, "main")
        self.ep = SectionModel(self, "plate")
        self.focus = "nav"
        self.nav = ui.Nav(DESIGN_TREE, title="DESIGN", state=self._state,
                          note=self._note, reason=self._reason)
        self.rank_list = ui.ListBox(title="RANKED ON THE MISSION")
        self.msg = ""
        self._solver_rows = None
        self._type_rows = None
        self._box_rows = None
        self._box_area = None
        self._conv_rows = None

    # -- opening --------------------------------------------------------------
    def open(self, key: str) -> None:
        self.key = key
        self.wing = Designer(self.g, key)
        for m in (self.af, self.ep):
            m.rebuild()
            m.ranked, m.clipped, m.refused, m.run = [], [], [], None
            #  the BAND belongs to a screen of THIS library on THIS mission,
            #  so opening a different slot must not inherit one: the composite
            #  would then be read on a map measured somewhere else.
            m.band = m.seed_sub = None
            m.fitted = m.declined = False
        self.af.seed_from(self.wing.spec.airfoil if self.wing.spec.airfoil in self.g.lib.airfoils
                          else sorted(self.g.lib.airfoils)[0])
        self.ep.seed_from(self.wing.spec.plate_airfoil or "naca0012"
                          if (self.wing.spec.plate_airfoil or "naca0012") in self.g.lib.airfoils
                          else sorted(self.g.lib.airfoils)[0])
        self._type_rows = self._box_rows = self._conv_rows = self._solver_rows = None
        self._box_area = None
        self.nav.refused = ""
        self.nav.select(self.nav.first_open(), force=True)
        self.focus = "nav"

    # -- what the navigator marks ---------------------------------------------
    # -- the gate ---------------------------------------------------------------
    def plate_locked(self) -> bool:
        """Is there a plate to design at all? AeroBO locks its own endplate
        stage on exactly this and says so: *"the plate has been taken off
        this design: its height is pinned at zero, so there is no surface
        here to give an aerofoil to"*. A LOCKED step is not a blocked one --
        no work upstream opens it, only raising the plate height does."""
        return bool(self.wing is None or self.wing.spec.plate_h <= 1e-9)

    def _state(self, key: str) -> str:
        grp, _, stage = key.partition(".")
        if grp == "af":
            return self.af.state(stage)
        if grp == "ep":
            if self.plate_locked():
                return "locked"
            if not self.af.finished():
                return "blocked"
            return self.ep.state(stage)
        #  the WING's four steps and the RESULTS' four are VIEWS OF ONE
        #  PROBLEM, not a procedure: a design box and a solver are two ways
        #  of looking at the same wing, and AeroBO's tab strip enables all of
        #  a stage's views together. So they are gated as a GROUP.
        if not self._wing_open():
            return "blocked"
        if grp == "w":
            if stage == "conv":
                return "done" if (self.wing and self.wing.result) else "ready"
            return "ready"
        return "ready" if (self.wing and self.wing.lap) else "blocked"

    def _wing_open(self) -> bool:
        """The wing group opens when both section questions are answered."""
        return bool(self.af.finished()
                    and (self.plate_locked() or self.ep.finished()))

    def _reason(self, key: str) -> str:
        grp, _, stage = key.partition(".")
        if grp == "af":
            return self.af.reason(stage)
        if grp == "ep":
            if self.plate_locked():
                return ("the end plates are at zero height, so there is no "
                        "surface here to give a section to. Raise 'end "
                        "plates' on the WING's design box and this opens")
            if not self.af.finished():
                return ("finish the AIRFOIL first: fit its section to the "
                        "wing with F. The plate is designed against the wing "
                        "the section is on, so the section has to be on it")
            return self.ep.reason(stage)
        if not self._wing_open():
            if not self.af.finished():
                return ("the wing flies a section, and this one has not been "
                        "chosen yet. Finish the AIRFOIL group -- screen, take "
                        "a winner, and fit it with F")
            return ("the end plates are part of the lattice this wing is "
                    "solved on. Design their section, or press ENTER on "
                    "ENDPLATE > library screening to fly them flat")
        if grp == "r" and not (self.wing and self.wing.lap):
            return "the lattice has not solved this wing"
        return ""

    def invalidate(self) -> None:
        """A form was rebuilt under us; drop anything caching its rows."""
        self._type_rows = self._box_rows = None
        self._conv_rows = self._solver_rows = None
        self._box_area = None

    def settle(self) -> None:
        """Keep the cursor on a step that is still open.

        A step can SHUT under the cursor -- raise the plates' height and the
        endplate group unlocks, drop it to zero while standing inside that
        group and there is nothing there any more. Called after anything that
        can move a gate."""
        if not self.nav.open(self.nav.current()):
            self.nav.select(self.nav.first_open(), force=True)

    def _note(self, key: str) -> str:
        if key.startswith(("af.", "ep.")):
            m = self._model(key)
            if key.endswith(".screen"):
                bad = m.dead_weighted()
                if bad:
                    return f"{', '.join(bad)} ranks nothing on a plate"
                return "" if m.ranked else "the weights decide the shortlist"
            if key.endswith(".rank") and not m.ranked:
                return "screen the library first (L)"
            if key.endswith(".opt") and scr.is_composite(m.objective) and not m.band:
                return "the band is measured by the screen"
        if key == "ep.section":
            return "a plate's camber is the whole effect"
        if key.startswith("r.") and not (self.wing and self.wing.lap):
            return "the lattice has not solved this wing"
        return ""

    def _model(self, key: str) -> SectionModel:
        return self.ep if key.startswith("ep.") else self.af

    # -- the rows the focused step owns ----------------------------------------
    def _wing_rows(self, keys, title_row):
        rows = [ui.Param(title_row[0], title_row[1], None, kind="label")]
        by = {p.key: p for p in self.wing.params.params}
        rows += [by[k] for k in keys if k in by]
        return rows

    def _n_strips(self, v):
        self.wing.spec.n_strips = int(min(max(int(v), 8), 64))
        self.wing.dirty = True
        self.wing.update()

    def rows(self):
        """The `ParamList` the selected step owns, or None."""
        k = self.nav.current()
        m = self._model(k)
        if k.endswith(".screen"):
            return m.screen_params
        if k.endswith(".section"):
            return m.params
        if k.endswith(".opt"):
            return m.opt_params
        if k == "w.type":
            if self._type_rows is None:
                self._type_rows = ui.ParamList(self._wing_rows(
                    self.TYPE_KEYS, ("t", "WING TYPE  (what is being built, and where)")))
            return self._type_rows
        if k == "w.box":
            #  the band rows follow the design vector, and the vector grows a
            #  row when the area is freed -- so the cache is keyed on that
            #  rather than built once
            if self._box_rows is None or self._box_area != self.wing.area_free:
                rows = self._wing_rows(
                    self.BOX_KEYS, ("b", "DESIGN BOX  (the vector, in order)"))
                rows += self.wing._box_rows()
                self._box_rows = ui.ParamList(rows)
                self._box_area = self.wing.area_free
            return self._box_rows
        if k == "w.solver":
            if self._solver_rows is None:
                P = ui.Param
                self._solver_rows = ui.ParamList([
                    P("s", "SOLVER", None, kind="label"),
                    P("lat", "3-D", lambda: "horseshoe vortex lattice", None,
                      kind="choice", choices=[], enabled=False,
                      help="imaged in a rigid wall at the ride height / standoff; "
                           "the tip plates are real panels, not a correlation"),
                    P("n", "strips across the span", lambda: self.wing.spec.n_strips,
                      self._n_strips, kind="int", lo=8, hi=64,
                      help="cosine-clustered. 24 is what every library wing was "
                           "analysed with; more costs the inner loop linearly"),
                    P("sec2d", "2-D", lambda: (self.wing.polar.source if self.wing.polar else "-"),
                      None, kind="choice", choices=[], enabled=False,
                      help="'xfoil' is measured; 'estimate' is the panel method's exact "
                           "inviscid slope plus CORRELATIONS for drag and stall"),
                    P("xf", "ask XFOIL for this section  (X)", None,
                      lambda _: self.wing.request_xfoil(), kind="action",
                      enabled=self.g.lib.use_xfoil),
                    P("p", "THE PLATE'S SECTION", None, kind="label"),
                    P("ps", "end plates fly",
                      lambda: (self.wing.spec.plate_airfoil or "a flat plate"),
                      None, kind="choice", choices=[], enabled=False,
                      help="set on WING TYPE, or designed by the ENDPLATE group and "
                           "fitted with the action below. Shown here because it is "
                           "part of the lattice this step describes"),
                    P("ph", "plate depth", lambda: self.wing.spec.plate_h, None,
                      lo=None, hi=None, unit="m", fmt="{:.3f}", enabled=False,
                      help="the plate's section barely moves the wing below about "
                           "0.10 m: there is too little panel for its camber to point. "
                           "It is the 'end plates' row of the DESIGN BOX"),
                    P("use", "fit the designed end plate  (ENTER)", None,
                      lambda _: self.accept_plate(), kind="action"),
                ] + self._wing_rows(self.SOLVER_KEYS,
                                    ("op", "OPTIMISER  (GP Bayesian, AeroBO)")))
            return self._solver_rows
        if k == "w.conv":
            if self._conv_rows is None:
                self._conv_rows = ui.ParamList(self._wing_rows(
                    self.CONV_KEYS, ("c", "CONVERGENCE  (keep going: ENTER)")))
            return self._conv_rows
        return None

    # -- actions ---------------------------------------------------------------
    def act(self) -> None:
        """ENTER on the selected step."""
        k = self.nav.current()
        rows = self.rows()
        p = rows.current() if rows is not None else None
        if p is not None and p.kind == "action" and self.focus == "rows":
            p.activate()
            return
        if k.endswith(".screen"):
            self._model(k).screen_library()

        elif k.endswith(".rank"):
            self._model(k).use_ranked(self.rank_list.idx)
        elif k.endswith(".opt"):
            self._model(k).optimise()
        elif k == "w.solver":
            self.g._run_optimiser()
        elif k == "w.conv":
            self.g._run_optimiser(extend=True)
        elif p is not None:
            p.activate()
        self.msg = self._model(k).msg if k[:2] in ("af", "ep") else (self.wing.msg if self.wing else "")

    def optimise(self, extend: bool = False) -> None:
        """What SQUARE / O does, whatever step is selected. `extend` is the
        wing's "keep going" (K, or ENTER on CONVERGENCE)."""
        k = self.nav.current()
        if k.startswith(("af.", "ep.")):
            self._model(k).optimise()
            self.msg = self._model(k).msg
        else:
            self.wing.optimise(extend=extend)
            self.msg = self.wing.msg

    def screen(self) -> None:
        """What L does -- on the group it belongs to, and nowhere else.

        It used to fall back to the AIRFOIL group from anywhere, so L pressed
        on a WING step screened a library the user was not looking at and
        jumped the navigator two groups back. A key that acts on something
        off screen is the same defect as a gate that lets you skip a step.
        """
        k = self.nav.current()
        if not k.startswith(("af.", "ep.")):
            self.msg = ("L screens a SECTION library -- select a step under "
                        "AIRFOIL or ENDPLATE first")
            return
        m = self._model(k)
        m.screen_library()
        self.msg = m.msg
        self.nav.select(k[:2] + ".rank")
        self.settle()

    def on_section_changed(self, m: "SectionModel") -> None:
        """A designed section is only real once the WING is flying it."""
        if self.wing is None or m.res.get("refused") or "coords" not in m.res:
            return
        if m is self.af:
            self.wing.update()

    def fit(self, m: "SectionModel") -> str:
        """Save a designed section into the library and put it on the wing."""
        return self._save(m, "sec" if m.target == "main" else "plate")

    def fit_and_advance(self, m: "SectionModel") -> str:
        """Fit, and then go where fitting just opened. ONE path, so the F key
        and the row's own action cannot end up doing different things -- the
        mouse walked the chain and stopped dead after fitting, because only
        the key had the second half."""
        name = self.fit(m)
        if name:
            self.advance("af" if m is self.af else "ep")
        return name

    def fit_current(self) -> str:
        """What F does: fit whichever section group is selected, and MOVE ON.

        Fitting is what finishes a group, so it is also what earns the step
        after it -- the navigator lands on the next open one rather than
        leaving the user to discover that something unlocked.
        """
        k = self.nav.current()
        if not k.startswith(("af.", "ep.")):
            self.msg = ("F fits a SECTION to the wing -- select a step under "
                        "AIRFOIL or ENDPLATE first")
            return ""
        return self.fit_and_advance(self._model(k))

    def advance(self, group: str) -> None:
        """Step to the first open node of the group AFTER `group`."""
        order = [g[:2] for g in ("af.", "ep.", "w.t", "r.s")]
        try:
            i = order.index(group[:2])
        except ValueError:
            return
        for nxt in order[i + 1:]:
            for key in self.nav.keys:
                if key.startswith(nxt) and self.nav.open(key):
                    self.nav.select(key)
                    return
        self.settle()

    def accept_plate(self) -> str:
        return self._save(self.ep, "plate")

    def _save(self, m: SectionModel, tag: str) -> str:
        if not m.res or m.res.get("refused") or "coords" not in m.res:
            self.g.hint = f"this section cannot be flown: {m.res.get('refused', 'not evaluated')}"
            return ""
        c = m.res["coords"]
        wu, wl = af.fit_cst(c, sec.N_CST)
        name = self.g.lib.unique_name("airfoils", f"{SLOT_ROLE[self.key]}-{tag}")
        try:
            self.g.lib.save_airfoil(af.AirfoilSpec(
                name=name, source="cst",
                w_upper=[float(v) for v in wu], w_lower=[float(v) for v in wl],
                notes=(f"designed on {self.g.mission.track} ({self.g.mission.surface}), "
                       f"{m.objective}, target {m.target}")))
        except (OSError, ValueError) as exc:
            self.msg = self.g.hint = _could_not_save(exc)
            return ""
        m.fitted = True
        if m is self.af:
            self.wing.spec.airfoil = name
            #  THE INCIDENCE COMES ACROSS ONLY IF THE OBJECTIVE PRICED IT.
            #  A composite run never builds the lattice, so its incidence
            #  coordinate is inert and the optimiser's choice of it is noise
            #  -- writing that onto the wing's mount angle is a design
            #  decision taken by a number nothing scored. See
            #  `SectionProblem.prices_inc`.
            if m.prob is not None and m.prob.prices_inc:
                lo, hi = BOUNDS[SLOT_ROLE[self.key]]["inc_deg"]
                self.g.build.slot(self.key).inc_deg = min(
                    max(float(m.res.get("inc_deg", 0.0)), lo), hi)
                self.g.build.sync_mirror(self.key)
        else:
            self.wing.spec.plate_airfoil = name
        self.wing.dirty = True
        self.wing.update()
        #  the section libraries these rows CYCLE are built once, at page
        #  open; a section designed since then would be shown but not be
        #  reachable by LEFT/RIGHT, which reads as the row having lost it.
        names = sorted(self.g.lib.airfoils)
        for pm in self.wing.params.params:
            if pm.key == "airfoil":
                pm.choices = list(names)
        self.g.section_for[self.key] = self.wing.spec.airfoil
        self.msg = self.g.hint = f"'{name}' saved and fitted to {self.wing.spec.name}"
        return name

    # -- drawing ----------------------------------------------------------------
    #: ranking columns: criterion -> (heading, how the RAW metric prints).
    #: The same table AeroBO's stage 2 draws, in the same order, and every
    #: column here is a criterion a weight prices -- the score is a weighted
    #: sum of exactly these, so a row that showed the metrics without the
    #: points would say what a section IS and nothing about where its score
    #: came from.
    RANK_COLUMNS = (("ldcr", "L/D@cl", "{:5.1f}"), ("clmax", "clmax", "{:4.2f}"),
                    ("ldmax", "L/Dmax", "{:5.1f}"), ("thick", "t/c", "{:5.3f}"),
                    ("astall", "astall", "{:+5.1f}"), ("cm", "|cm|", "{:5.3f}"),
                    ("cdcr", "cd@cl", "{:6.4f}"))

    def refresh_rank(self) -> None:
        m = self._model(self.nav.current())
        w = scr.normalised(m.weights)
        items = []
        for name, score, res, _ in m.ranked[:60]:
            if res.get("refused") and res.get("metrics") is None:
                sub, val = res["refused"][:70], "refused"
            else:
                met = res.get("metrics") or {}
                pts = res.get("points") or {}
                bits = []
                #  COLUMNS IN WEIGHT ORDER, heaviest first. The row is wider
                #  than the panel with all seven on it, so something is cut --
                #  and what should survive the cut is what the score is mostly
                #  made of. A criterion at zero weight goes last, which is
                #  where it belongs on a table that exists to say where a
                #  score came from.
                cols = sorted(self.RANK_COLUMNS,
                              key=lambda t: (-w.get(t[0], 0.0),
                                             self.RANK_COLUMNS.index(t)))
                for k, head, fmt in cols:
                    if k not in met:
                        continue
                    #  the metric, and faint beside it what it PUT INTO the
                    #  score. A criterion nobody weighted shows the number and
                    #  no points: "nobody can rank this" and "you set this to
                    #  zero" are different sentences and the table makes both.
                    cell = f"{head} " + fmt.format(met[k])
                    if w.get(k, 0.0) > 0.0 and k in pts:
                        cell += f"({pts[k]:+.1f})"
                    bits.append(cell)
                sub = "  ".join(bits)
                lap = res.get("lap")
                if lap is not None and res.get("d_lap") is not None:
                    sub += f"   lap {res['d_lap']:+.3f} s"
                val = f"{score:.2f} pts" if math.isfinite(score) else "refused"
            if name in m.clipped:
                sub += "  [CLIPPED]"
            items.append((name, sub, val))
        for name, why in m.refused[:20]:
            items.append((name, f"refused: {why}", "-"))
        self.rank_list.set_items(items, keep=m.seed)

    def draw(self, screen, text: ui.Text, plot: ui.Plot, u: float) -> str:
        R = lambda x, y, w, h: (int(x * u), int(y * u), int(w * u), int(h * u))   # noqa: E731
        self.nav.draw(screen, text, ui.panel(screen, R(12, 12, 300, 664)),
                      row_h=int(22 * u), size=13, focus=(self.focus == "nav"))
        k = self.nav.current()
        rows = self.rows()
        help_ = ""
        if k.endswith(".rank"):
            self.refresh_rank()
            self.rank_list.draw(screen, text, ui.panel(screen, R(324, 12, 620, 664)),
                                row_h=int(33 * u), size=13, focus=(self.focus == "rows"))
            body = R(956, 12, 312, 664)
        elif rows is not None:
            help_ = rows.draw(screen, text, ui.panel(screen, R(324, 12, 400, 664)),
                              row_h=int(21 * u), size=13, focus=(self.focus == "rows"))
            body = R(736, 12, 532, 664)
        else:
            body = R(324, 12, 944, 664)

        grp = k.split(".")[0]
        if grp in ("af", "ep"):
            self._draw_section(screen, text, plot, u, body, self._model(k), k)
        elif grp == "w":
            self._draw_wing(screen, text, plot, u, body, k)
        else:
            self._draw_results(screen, text, plot, u, body, k)
        return help_

    # -- the two section groups --------------------------------------------------
    def _draw_section(self, screen, text, plot, u, body, m, k) -> None:
        bx, by, bw, bh = body
        stage = k.split(".")[1]
        if stage == "screen":
            self._draw_screen(screen, text, u, body, m)
            return
        if stage == "rank":
            r = ui.panel(screen, (bx, by, bw, 300))
            i = self.rank_list.idx
            if 0 <= i < len(m.ranked):
                res = m.ranked[i][2]
                c = res.get("coords")
                if c is not None:
                    ym = float(np.max(np.abs(c[:, 1]))) * 1.35 + 0.01
                    plot.begin(screen, r, (-0.02, 1.02), (-ym, ym),
                               title=f"{m.ranked[i][0]}", equal=True)
                    plot.line(c[:, 0], c[:, 1], col=ui.C_ACCENT, width=2, closed=True)
            r2 = ui.panel(screen, (bx, by + 310, bw, bh - 310))
            x, y = r2.x + 10, r2.y + 10
            for ln in ("THE SHORTLIST IS CHOSEN ON THE",
                       "MAP IT IS JUDGED ON.",
                       "",
                       "Every score here is the weighted sum of",
                       "the seven criteria you set on the",
                       "SCREENING step, read on a 0-100 band",
                       "measured over this library. The three",
                       "composite objectives maximise the SAME",
                       "number, so the section the search",
                       "starts from and the section it is",
                       "trying to beat are on one scale.",
                       "",
                       "The figure in brackets is what that",
                       "criterion PUT INTO the score. A column",
                       "with no bracket carries no weight.",
                       "",
                       "ENTER loads the highlighted section",
                       "into the SHAPE -- and makes it the seed",
                       "the goal objective holds as a floor."):
                text.blit(screen, ln, x, y, 11, ui.C_DIM)
                y += 15
            return

        # shape
        r1 = ui.panel(screen, (bx, by, bw, 228))
        c = m.res.get("coords")
        if c is not None:
            ym = float(np.max(np.abs(c[:, 1]))) * 1.35 + 0.01
            ttl = ("END PLATE SECTION" if m.target == "plate" else "SECTION")
            plot.begin(screen, r1, (-0.02, 1.02), (-ym, ym),
                       title=f"{ttl}   {m.seed} -> designed", equal=True)
            plot.line(c[:, 0], c[:, 1], col=ui.C_ACCENT, width=2, closed=True)
            try:
                c0 = self.g.lib.airfoils[m.seed].coords()
                plot.line(c0[:, 0], c0[:, 1], col=ui.C_DIM, width=1, closed=True)
            except Exception:
                pass
        # polar
        pol = m.res.get("polar")
        r2 = ui.panel(screen, (bx, by + 238, bw, 206))
        if pol is not None:
            plot.begin(screen, r2, (float(pol.alpha.min()), float(pol.alpha.max())),
                       (float(pol.cl.min()), float(pol.cl.max())),
                       title=f"cl vs alpha   [{pol.source.upper()}]", xlabel="deg")
            plot.line(pol.alpha, pol.cl, col=ui.C_ACCENT)
            plot.vline(float(m.res.get("alpha_L0_deg", 0.0)), col=ui.C_KEY)
        # the run / the verdict
        r3 = ui.panel(screen, (bx, by + 454, bw, bh - 454))
        x, y = r3.x + 10, r3.y + 10
        if m.res.get("refused"):
            text.blit(screen, "REFUSED", x, y, 13, ui.C_WARN, bold=True)
            y += 18
            text.blit(screen, m.res["refused"][:56], x, y, 11, ui.C_WARN)
            y += 20
        elif "lap" in m.res:
            d = float(m.res.get("d_lap", 0.0))
            text.blit(screen, f"lap {m.res['lap'].time:.4f} s", x, y, 14,
                      ui.C_OK if d < 0 else ui.C_WARN, bold=True)
            y += 20
            text.blit(screen, f"{d:+.4f} s vs the car without this wing", x, y, 12,
                      ui.C_OK if d < 0 else ui.C_WARN)
            y += 20
        if stage == "opt" and m.run:
            tr = np.asarray(m.run["trace"], dtype=float)
            fin = tr[np.isfinite(tr)]
            if fin.size > 1:
                plot.begin(screen, (x, y, bw - 20, bh - (y - by) - 34),
                           (0, len(tr)), (float(fin.min()), float(fin.max())),
                           title="best so far", xlabel="evaluation")
                plot.line(np.arange(len(tr)), tr, col=ui.C_ACCENT)
                rt = np.asarray(m.run["rs_trace"], dtype=float)
                if rt.size == tr.size:
                    plot.line(np.arange(len(rt)), rt, col=ui.C_DIM, width=1)
        elif m.msg:
            text.blit(screen, m.msg[:58], x, y, 11, ui.C_KEY)

    def _draw_screen(self, screen, text, u, body, m) -> None:
        """WHAT THE SCORE IS MADE OF: one bar per criterion, at the weight it
        carries, and -- once the library has been screened -- the points the
        current winner actually took from it. The bars are the same numbers
        the rows beside them hold, drawn because a set of seven fractions is a
        shape before it is a table."""
        import pygame as pg
        bx, by, bw, bh = body
        r = ui.panel(screen, (bx, by, bw, 300))
        w = scr.normalised(m.weights)
        dead = m.dead()
        best = m.ranked[0][2] if m.ranked else None
        pts = (best or {}).get("points") or {}
        x0, y = r.x + 12, r.y + 12
        text.blit(screen, "THE SCORE, BY CRITERION", x0, y, 13, ui.C_SECTION, bold=True)
        y += 22
        bar_x, bar_w = x0 + 96, r.width - 96 - 140
        for c in scr.CRITERIA:
            col = ui.C_DIM if (c in dead or w[c] <= 0.0) else ui.C_TEXT
            text.blit(screen, scr.SHORT[c], x0, y, 11, col)
            pg.draw.rect(screen, ui.C_SEL_BG, (bar_x, y + 1, bar_w, 11))
            if w[c] > 0.0:
                pg.draw.rect(screen, ui.C_DIM if c in dead else ui.C_ACCENT,
                             (bar_x, y + 1, max(2, int(bar_w * w[c] / 0.45)), 11))
            txt = f"{100 * w[c]:4.0f} %"
            if c in pts and w[c] > 0.0:
                txt += f"   {pts[c]:+5.1f} pts"
            elif c in dead and w[c] > 0.0:
                txt = "ranks nothing"
            text.blit(screen, txt, r.right - 12, y, 11, col, right=True)
            y += 17
        y += 8
        if best is not None:
            text.blit(screen, f"'{m.ranked[0][0]}' scores "
                              f"{m.ranked[0][1]:.2f} of a possible 100",
                      x0, y, 12, ui.C_OK)
            y += 18
            text.blit(screen, f"{len(m.ranked)} ranked, {len(m.refused)} refused by a gate",
                      x0, y, 11, ui.C_DIM)
        else:
            text.blit(screen, "press L to screen the library on these weights",
                      x0, y, 12, ui.C_KEY)

        r2 = ui.panel(screen, (bx, by + 310, bw, bh - 310))
        x, y = r2.x + 10, r2.y + 10
        bad = m.dead_weighted()
        lines = ["THE WEIGHTS ARE THE QUESTION.", ""] + _wrap(
                    "These sliders say what matters to you - grip, efficiency, stall "
                    "safety. The library is ranked by them.", 42) + ["",
                 f"judged at cl {m.cl_design:.3f}" +
                 ("   (an end plate carries no design load)"
                  if m.target == "plate" else ""), ""]
        if bad:
            lines += ["WARNING", ""] + _wrap(scr.DEAD[m.target][bad[0]], 44)
        elif m.target == "plate":
            lines += _wrap(scr.DEAD["plate"]["ldcr"], 44)
        for ln in lines:
            text.blit(screen, ln, x, y, 11,
                      ui.C_WARN if bad and ln == "WARNING" else ui.C_DIM)
            y += 15
            if y > r2.bottom - 16:
                break

    # -- the wing group ----------------------------------------------------------
    def _draw_wing(self, screen, text, plot, u, body, k) -> None:
        bx, by, bw, bh = body
        w = self.wing
        spec, a = w.spec, w.spec.aero
        if k == "w.conv" and w.result:
            r = ui.panel(screen, (bx, by, bw, 300))
            rr = w.result
            tr = np.asarray([v if math.isfinite(v) else np.nan for v in rr["trace"]])
            fin = tr[np.isfinite(tr)]
            if fin.size > 1:
                plot.begin(screen, r, (0, len(tr)), (float(fin.min()), float(fin.max())),
                           title=f"best so far   {rr['objective']}", xlabel="evaluation")
                plot.line(np.arange(len(tr)), tr, col=ui.C_ACCENT)
                rt = np.asarray([v if math.isfinite(v) else np.nan for v in rr["rs_trace"]])
                if rt.size == tr.size:
                    plot.line(np.arange(len(rt)), rt, col=ui.C_DIM, width=1)
                #  the graph does not reset on "keep going": the run being
                #  continued is the left of the same curve, and the mark is
                #  where the continuation picked up
                if rr.get("n_prior", 0) > 0:
                    plot.vline(float(rr["n_prior"]), col=ui.C_DIM)
            r2 = ui.panel(screen, (bx, by + 310, bw, bh - 310))
            x, y = r2.x + 10, r2.y + 10
            fmt = ((lambda v: f"{-v:.4f} s") if rr["objective"] == "lap time"
                   else (lambda v: f"{v:.4g}"))
            n_ln = (f"{rr['n']} evaluations in {rr['secs']:.1f} s"
                    + (f"   (continued from {rr['n_prior']})" if rr.get("n_prior", 0) else ""))
            for ln in (f"BO      {fmt(rr['bo'])}",
                       f"random  {fmt(rr['rs'])}   (same budget)",
                       f"start   {fmt(rr['start'])}",
                       n_ln,
                       "", rr.get("design", "")):
                text.blit(screen, ln, x, y, 12, ui.C_TEXT)
                y += 17
            return
        # planform
        r1 = ui.panel(screen, (bx, by, bw, 300))
        #  THE SPAN THE WING FLIES, which is the row only while the plates are
        #  bolted on square. A blended plate leans outboard and the wing gives
        #  that up, so drawing the row here would draw a wing nobody flew.
        b_fl = float(a.get("span_flown", spec.span)) or spec.span
        reach = float(a.get("plate_projection", 0.0))
        b2 = 0.5 * b_fl
        etas = np.linspace(-1.0, 1.0, 41)
        cs = spec.chord * (1.0 - (1.0 - spec.taper) * np.abs(etas))
        c_tip = spec.chord * spec.taper
        plot.begin(screen, r1, (-b2 - reach - 0.1, b2 + reach + 0.1),
                   (-0.6 * spec.chord - 0.15, 0.6 * spec.chord + 0.1),
                   title=f"planform   {spec.name}   S {spec.S:.3f} m2  AR {spec.AR:.2f}"
                         + (f"   span {b_fl:.3f} of {spec.span:.3f} m" if reach > 1e-9 else ""),
                   equal=True)
        xs = np.concatenate([etas * b2, (etas * b2)[::-1]])
        ys = np.concatenate([0.5 * cs, -0.5 * cs[::-1]])
        plot.fill(xs, ys, (70, 45, 20))
        plot.line(xs, ys, C_PANEL_ON, 2, closed=True)
        if spec.plate_h > 0:
            for sgn in (-1, 1):
                plot.line([sgn * b2, sgn * b2], [-0.65 * c_tip, 0.65 * c_tip], C_PLATE, 3)
                if reach > 1e-9:
                    #  seen from above, a blended plate is a ribbon reaching
                    #  outboard: the wing ends at b2 and the plate's TIP is out
                    #  here, which is the span the row gave up
                    plot.line([sgn * (b2 + reach)] * 2, [-0.55 * c_tip, 0.55 * c_tip],
                              C_PLATE, 2)
                    plot.line([sgn * b2, sgn * (b2 + reach)], [0.6 * c_tip, 0.55 * c_tip],
                              C_PLATE, 1)
                    plot.line([sgn * b2, sgn * (b2 + reach)], [-0.6 * c_tip, -0.55 * c_tip],
                              C_PLATE, 1)
        # the laws
        r2 = ui.panel(screen, (bx, by + 310, bw, bh - 310))
        x, y = r2.x + 10, r2.y + 10
        if w.err:
            text.blit(screen, f"lattice refused: {w.err[:48]}", x, y, 12, ui.C_WARN)
            y += 18
        if "CLa" in a:
            for ln in (f"CL  = {a['CL0']:+.3f} + {a['CLa']:.3f} alpha",
                       f"      clamped [{a['CL_min']:+.2f}, {a['CL_max']:+.2f}]",
                       f"CD  = {a['cd0']:.4f} {a['cd1']:+.4f} CL {a['cd2']:+.4f} CL^2",
                       f"e {a['e']:.3f}   stall {a['alpha_stall_deg']:+.1f} deg",
                       f"section {spec.airfoil}",
                       f"plates  {spec.plate_airfoil or 'flat'}"
                       + (f", blend {spec.plate_blend:.2f} {spec.plate_shape}"
                          if spec.plate_blend > 0.0 else "")):
                text.blit(screen, ln, x, y, 12, ui.C_TEXT)
                y += 17
        if w.lap is not None and w.lap.ok:
            d = w.lap.time - w.base_lap.time
            y += 6
            text.blit(screen, f"lap {w.lap.time:.4f} s   {d:+.4f} s", x, y, 13,
                      ui.C_OK if d < 0 else ui.C_WARN, bold=True)

    # -- results -----------------------------------------------------------------
    def _draw_results(self, screen, text, plot, u, body, k) -> None:
        bx, by, bw, bh = body
        w = self.wing
        spec, a = w.spec, w.spec.aero
        r = ui.panel(screen, (bx, by, bw, bh))
        x, y = r.x + 14, r.y + 12
        stage = k.split(".")[1]

        def line(s, col=ui.C_TEXT, size=13, dy=19):
            nonlocal y
            text.blit(screen, s, x, y, size, col)
            y += dy

        if stage == "summary":
            line(f"MISSION   {self.g.mission.track} ({self.g.mission.surface})",
                 ui.C_SECTION, 14, 24)
            if w.base_lap.ok:
                line(f"  the car with this slot EMPTY      {w.base_lap.time:.4f} s")
            if w.lap is not None and w.lap.ok:
                d = w.lap.time - w.base_lap.time
                line(f"  with the wing designed here       {w.lap.time:.4f} s   {d:+.4f} s",
                     ui.C_OK if d < 0 else ui.C_WARN)
            y += 10
            line("AIRFOIL", ui.C_SECTION, 14, 22)
            line(f"  screened {len(self.af.ranked)} library sections"
                 + (f", {len(self.af.clipped)} clipped" if self.af.clipped else ""))
            line(f"  section   {spec.airfoil}    t/c {self.af.res.get('tc', 0):.4f}"
                 f"   cl_max {self.af.res.get('cl_max', 0):.3f}")
            line(f"  optimiser {'run' if self.af.run else 'not run'}"
                 + (f", {self.af.run['n']} evals" if self.af.run else ""))
            y += 10
            line("ENDPLATE", ui.C_SECTION, 14, 22)
            line(f"  screened {len(self.ep.ranked)} library sections")
            line(f"  section   {spec.plate_airfoil or 'a flat plate (the lattice default)'}")
            line(f"  height    {spec.plate_h:.3f} m")
            y += 10
            line("WING", ui.C_SECTION, 14, 22)
            line(f"  {spec.name}   S {spec.S:.3f} m2   AR {spec.AR:.2f}   "
                 f"span {spec.span:.3f} m   chord {spec.chord:.3f} m")
            line(f"  mount {spec.mount}   incidence {w.slot.inc_deg:+.1f} deg   "
                 f"{'ride' if w.role == 'top' else 'standoff'} {spec.ride_h_flown:.2f} m")
            if w.result:
                line(f"  optimiser run, {w.result['n']} evals")
            y += 10
            est = a.get("polar_is_estimate", True)
            line("the wing is flying an ESTIMATE polar" if est
                 else "the wing is flying an XFOIL-measured polar",
                 ui.C_WARN if est else ui.C_OK, 12)
            line("cl_max and the drag bucket are correlations there, +-15 % on drag;"
                 if est else
                 "measured. But the SEARCH above did not use it:", ui.C_DIM, 11, 15)
            if est:
                line("the lift slope and zero-lift angle are the panel method's,",
                     ui.C_DIM, 11, 15)
                line("which is exact for what it models.", ui.C_DIM, 11, 15)
            else:
                #  worth saying plainly, because it is the one place the two
                #  stages are measured on different instruments.
                line("every candidate in the SCREEN and the SHAPE OPTIMISATION was",
                     ui.C_DIM, 11, 15)
                line("scored on `polar.estimate_polar` -- 3 ms against XFOIL's ~2.4 s,",
                     ui.C_DIM, 11, 15)
                line("which is the only reason a 48-evaluation search is affordable.",
                     ui.C_DIM, 11, 15)
                line("So a section that wins the 2-D screen need not win on XFOIL.",
                     ui.C_WARN, 11, 15)
                line("X on the SOLVER step measures the winner properly.", ui.C_DIM, 11, 15)
        elif stage == "geometry":
            line("GEOMETRY", ui.C_SECTION, 14, 24)
            for lab, v, un in (("span b", spec.span, "m"), ("root chord", spec.chord, "m"),
                               ("taper", spec.taper, ""), ("reference area S", spec.S, "m2"),
                               ("aspect ratio", spec.AR, ""), ("MAC", spec.mac, "m"),
                               ("root twist", spec.twist_root_deg, "deg"),
                               ("tip twist", spec.twist_deg, "deg"),
                               ("end plate height", spec.plate_h, "m"),
                               ("image-plane gap", spec.ride_h_flown, "m"),
                               ("Reynolds", a.get("Re", 0.0), "")):
                line(f"  {lab:<22s} {v:10.4f} {un}")
            y += 8
            line("  chord = 2S / b(1 + taper)  -- derived, not a row", ui.C_DIM, 11, 16)
            line(f"  mass  {wing_mass(spec, DEV_OUT0 + DEV_OUT1 if w.role == 'flank' else TOP_PYLON_L):.2f} kg"
                 f"  (two skins, two plates, two mounts; no ribs or fasteners)", ui.C_DIM, 11)
        elif stage == "loading":
            sd = w.span_data
            if sd is not None and w.polar is not None:
                cl = np.asarray(sd["cl"])
                b_fl = float(a.get("span_flown", spec.span)) or spec.span
                eta = np.asarray(sd["y"]) / max(0.5 * b_fl, 1e-6)
                ymax = max(1.2, float(np.max(np.abs(cl))) * 1.15, w.polar.cl_max * 1.05)
                plot.begin(screen, (bx, by, bw, 320), (-1.0, 1.0),
                           (min(0.0, float(cl.min()) * 1.2), ymax),
                           title="strip cl across the span, at the mount incidence",
                           xlabel="y / (b/2)")
                plot.hline(w.polar.cl_max, ui.C_WARN)
                plot.label(0.30, w.polar.cl_max, "section cl_max", ui.C_WARN, 10, dy=3)
                plot.line(eta, cl, C_PANEL_ON, 2)
                plot.line(eta, np.asarray(sd["aeff"]) / 20.0, ui.C_LINE4, 1)
                y = by + 336
                line(f"CL {sd['CL']:.4f}    span efficiency e {sd['e']:.4f}", ui.C_TEXT, 13, 20)
                line(f"peak strip cl {float(np.max(cl)):.3f} against a section cl_max "
                     f"of {w.polar.cl_max:.3f}", ui.C_DIM, 12, 18)
                line("blue: the strip's effective incidence / 20", ui.C_LINE4, 11)
                yp = np.asarray(sd.get("y_plate", []), float)
                if yp.size:
                    #  THE JUNCTION ITSELF, seen from behind -- the one view the
                    #  planform cannot show, because the transition happens out
                    #  of the planform's plane. A sharp corner draws a right
                    #  angle here; a blend draws the curve the lattice flew.
                    zp = np.asarray(sd["z_plate"], float)
                    half = 0.5 * b_fl
                    zmax = max(float(np.max(np.abs(zp))), 1e-3)
                    plot.begin(screen, (bx, y + 6, bw, 190),
                               (-half - 1.25 * zmax, half + 1.25 * zmax),
                               (-0.25 * zmax, 1.25 * zmax),
                               title=("wing and plate, seen from behind   "
                                      + (f"blend {spec.plate_blend:.2f} "
                                         f"{spec.plate_shape}" if spec.plate_blend > 0
                                         else "square corner (blend 0)")),
                               equal=True)
                    plot.line([-half, half], [0.0, 0.0], C_PANEL_ON, 3)
                    for sgn in (-1.0, 1.0):
                        side = yp * sgn > 0.0
                        o = np.argsort(np.abs(yp[side]))
                        plot.line(np.concatenate([[sgn * half], yp[side][o]]),
                                  np.concatenate([[0.0], zp[side][o]]), C_PLATE, 3)
                    y += 200
            else:
                line("the lattice has not solved this wing", ui.C_WARN)
        else:   # evaluations
            line("EVALUATIONS", ui.C_SECTION, 14, 24)
            for nm, run, model in (("airfoil", self.af.run, self.af),
                                   ("end plate", self.ep.run, self.ep),
                                   ("wing", w.result, None)):
                if not run:
                    line(f"  {nm:<11s} not run", ui.C_DIM)
                    continue
                if model is not None:
                    line(f"  {nm:<11s} {run['n']:3d} evals   BO {model.fmt_score(run['bo'])}"
                         f"   random {model.fmt_score(run['rs'])}"
                         f"   start {model.fmt_score(run['start'])}   {run['secs']:.1f} s")
                else:
                    f = ((lambda v: f"{-v:.4f} s") if run["objective"] == "lap time"
                         else (lambda v: f"{v:.4g}"))
                    line(f"  {nm:<11s} {run['n']:3d} evals   BO {f(run['bo'])}"
                         f"   random {f(run['rs'])}   start {f(run['start'])}   {run['secs']:.1f} s")
            y += 12
            line("The random twin runs the SAME budget alongside every search.",
                 ui.C_DIM, 11, 16)
            line("It is what says whether the GP bought anything -- AeroBO's", ui.C_DIM, 11, 16)
            line("crossover-map question, at garage scale.", ui.C_DIM, 11, 16)
            if self.af.clipped:
                y += 10
                line(f"clipped into the box during screening ({len(self.af.clipped)}):",
                     ui.C_WARN, 12, 17)
                line("  " + ", ".join(self.af.clipped[:6])
                     + (" ..." if len(self.af.clipped) > 6 else ""), ui.C_DIM, 11)


class Garage:
    """Owns the window, the build, the library and the input. `run()`
    returns 'drive' or 'quit'; the (clamped) build is `self.build` either
    way (`self.design` is the same object, for the old attribute name)."""

    PAD_MOVE_X = 0.9      # m/s of x at full stick
    PAD_MOVE_H = 0.6      # m/s of h at full stick
    PAD_ORBIT = 2.2       # rad/s at full stick
    PAD_REPEAT = 0.14     # s between stick-held adjustments on the list pages

    def __init__(self, size=(1280, 800), build=None, pad=None, headless: bool = False,
                 lib: Library | None = None):
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("carsim - garage")
        self.lib = lib or library()
        if isinstance(build, WingDesign):
            build = CarBuild.from_json(asdict(build))
        self.build: CarBuild = (build or CarBuild.load() or CarBuild()).clamp(self.lib)
        self.view = GarageView(self.screen)
        self.cam = Orbit(*self.screen.get_size())
        self.text = self.view.text
        self.plot = ui.Plot(self.text)
        self.pad = pad
        self.headless = headless
        self.deploy_cmd = 0.0
        self.deploy = 0.0
        self.hint = ""
        self._hint_t = 0.0
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
        self.designer: Designer | None = None
        #  the three design steps, in the order AeroBO walks them. The mission
        #  is held on the GARAGE and not on a page, because the section and the
        #  wing are both scored against it and a mission that lived on one page
        #  could be changed underneath the other.
        self.mission = ms.MissionSpec()
        self.section_for: dict[str, str] = {}      # slot -> the section designed for it
        self.mission_page = MissionPage(self)
        self.design_page = DesignPage(self)
        self.af_page = AirfoilPage(self)
        self.lib_page = LibraryPage(self)
        self.prompt = ui.TextPrompt()
        self._prompt_kind = ""
        self._af_return = "car"
        self.status = ""
        #  the wing-design tutorial (drive/wing_tutorial.py, task 24): a
        #  WingTutor the frame and the menu query, or None; `progress` is the
        #  player's runs/progress.json (drive/progress.py), None in a script
        self.tutor = None
        self.progress = None

    # the old attribute name
    @property
    def design(self) -> CarBuild:
        return self.build

    def set_paint(self, rgb) -> None:
        """The player's paint on the preview car (GarageView.set_paint):
        drive.drive calls it right after constructing the Garage, with the
        fitted car's paint resolved to an RGB. Cosmetic only."""
        self.view.set_paint(rgb)

    # -- pause menu (ESC / OPTIONS) -------------------------------------------
    def _menu_open(self) -> None:
        secs = [("KEYBOARD", GARAGE_HELP_KB)]
        note = ""
        if self.pad is not None:
            secs.append(("PS5 DUALSENSE" if self.pad.layout == "ps" else "GAMEPAD",
                         GARAGE_HELP_PAD))
        else:
            note = "no controller: pair the DualSense (CREATE+PS) - it hot-plugs"
        tut_rows = []
        if self.tutor is not None and self.tutor.active:
            tut_rows = self.tutor.menu_rows()
        elif self.progress is not None:
            from .wing_tutorial import menu_row
            tut_rows = [menu_row(self.progress, self.tutor)]
        self.menu.show(
            items=[("Resume", "resume"),
                   (f"Design the {self.sel} wing  (mission -> section -> wing)", "design")]
                  + tut_rows
                  + [("Airfoil library", "airfoils"),
                     ("Wing & build library", "library"),
                     ("Reset car to defaults", "defaults"),
                     ("Reset camera", "camera"),
                     ("Drive this car", "drive"),
                     ("Quit", "quit")],
            sections=secs, note=note, title="GARAGE",
            subtitle=self.build.summary(self.lib)[:120],
            footer="ESC / OPTIONS resume   R defaults   C camera   ENTER / CROSS select")
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
        self.menu.show(items=[("Guided first wing (recommended)", "wt_start"),
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
        """Run a menu action; returns 'drive' / 'quit' for the loop, else None."""
        if action in (None, "resume"):
            return None
        if action == "defaults":
            self.build.reset()
            self.build.clamp(self.lib)
            self.designer = None
            self.hint = "car reset: no wings"
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
        elif action in ("wt_start", "wt_resume"):
            self._first_wing_answered()
            from .wing_tutorial import WingTutor, saved_state
            start = saved_state(self.progress)["step"] if action == "wt_resume" else None
            self.tutor = WingTutor(self.progress, start=start)
            self.hint = f"wing tutorial {self.tutor.label()}: {self.tutor.step.title}"
        elif self.tutor is not None and self.tutor.menu_action(action):
            self.hint = "wing tutorial: " + ("ended" if not self.tutor.active
                                             else self.tutor.step.title)
        elif action in ("drive", "quit"):
            return action
        return None

    # -- pages ------------------------------------------------------------------
    # -- the three design steps, in order ------------------------------------
    def open_mission(self, key: str | None = None) -> None:
        """STEP 1. Every route into designing a wing comes through here."""
        if key is not None:
            self.sel = key
        if self._first_wing_choice():
            return
        self.mission_page.update()
        self.page = "mission"

    def open_section(self, key: str | None = None) -> None:
        """STEP 2. The DESIGN page -- the navigator over everything downstream
        of the mission. Gated on a STATED mission."""
        if key is not None:
            self.sel = key
        if not self.mission.stated:
            self.hint = "state the mission first: nothing downstream can be judged without one"
            self.open_mission()
            return
        self.design_page.open(self.sel)
        self.page = "section"

    def open_designer(self, key: str | None = None, airfoil: str | None = None,
                      inc_deg: float | None = None) -> None:
        """STEP 3. Gated on a stated mission AND a section designed for this
        slot -- AeroBO's order, enforced rather than suggested. `airfoil` and
        `inc_deg` are what step 2 hands over: the section it designed, and the
        rigging angle it designed it at, which SEEDS the wing's own incidence
        row over the same band."""
        if key is not None:
            self.sel = key
        if not self.mission.stated:
            self.hint = "state the mission first (step 1 of 2)"
            self.open_mission()
            return
        if airfoil is None and self.sel not in self.section_for:
            self.hint = "design the section first (step 2 of 2)"
            self.open_section()
            return
        if self.design_page.wing is None or self.design_page.key != self.sel:
            self.design_page.open(self.sel)
        self.designer = self.design_page.wing
        if airfoil:
            self.designer.spec.airfoil = airfoil
            self.designer.dirty = True
        if inc_deg is not None:
            lo, hi = BOUNDS[SLOT_ROLE[self.sel]]["inc_deg"]
            self.build.slot(self.sel).inc_deg = min(max(float(inc_deg), lo), hi)
            self.build.sync_mirror(self.sel)
        if airfoil or inc_deg is not None:
            self.designer.update()
        #  there is no separate wing PAGE any more: the wing is a group of the
        #  design navigator, so "open the designer" lands on its design box --
        #  IF the gate has opened it. On a slot whose section has not been
        #  chosen yet it lands on the first step that is open instead, which
        #  is where the work actually starts.
        dp = self.design_page
        if not dp.nav.select("w.box"):
            self.hint = dp.nav.refused or self.hint
            dp.nav.select(dp.nav.first_open(), force=True)
            dp.focus = "nav"
        else:
            dp.focus = "rows"
        self.page = "section"

    def open_airfoils(self) -> None:
        """The AIRFOIL LIBRARY: browse and rank the 34 shipped sections. NOT
        the section DESIGNER (step 2) -- this page picks one that exists, that
        page makes a new one. Reachable from anywhere, and NOT gated: reading
        the library is not designing a wing.

        With no designer open it is opened against the wing in the selected
        slot, or against the role's defaults if that slot is empty. It used to
        call `open_designer` here, which now diverts into the mission page and
        would have left `d` None one line later."""
        d = self.designer
        if d is None and self.page == "section":
            d = self.designer = self.design_page.wing
        if d is not None:
            self._af_return = "section" if self.page == "section" else "car"
            cl = d.dp.get("CL", 0.8) if d.dp else 0.8
            self.af_page.open(d.spec.reynolds(), cl_design=abs(cl) + 0.2, keep=d.spec.airfoil)
        else:
            self._af_return = "car"
            spec = self.lib.wings.get(self.build.slot(self.sel).wing)
            re = spec.reynolds() if spec is not None else 5e5
            self.af_page.open(re, cl_design=1.0,
                              keep=spec.airfoil if spec is not None else None)
        self.page = "airfoil"

    def open_library(self) -> None:
        self.lib_page.refresh()
        self.page = "library"

    def close_page(self) -> None:
        """ESC steps BACK through the design pages rather than dropping
        straight to the car: the chain is the procedure, and walking out of
        the middle of it is how someone ends up with a wing designed against a
        mission they never looked at."""
        if self.page == "section" and self.design_page.wing is not None \
                and self.design_page.wing.dirty:
            self.designer = self.design_page.wing
            self.designer.commit()
            self.hint = self.designer.msg
        if self.page == "airfoil":
            self.page = self._af_return
            return
        back = {"section": "mission", "mission": "car"}
        self.page = back.get(self.page, "car")
        if self.page == "mission":
            self.mission_page.update()

    def assign_airfoil(self) -> None:
        name = self.af_page.current_name()
        if name and self.designer is not None:
            self.designer.spec.airfoil = name
            self.designer.dirty = True
            self.designer.update()
            self.hint = f"section {name} on {self.designer.spec.name}"
            self.page = self._af_return if self._af_return in PAGES else "section"
            if self.page == "section":
                #  the AIRFOIL group is now showing a section the wing is not
                #  flying, so re-seed it from the one that was just assigned.
                self.design_page.af.rebuild()
                self.design_page.af.seed_from(name)

    def prompt_rename(self) -> None:
        if self.designer is None:
            return
        self._prompt_kind = "rename"
        self.prompt.show("wing name", self.designer.spec.name)

    def prompt_naca(self) -> None:
        self._prompt_kind = "naca"
        self.prompt.show("new NACA 4-digit section (e.g. 4415)", "")

    def prompt_build(self) -> None:
        self._prompt_kind = "build"
        self.prompt.show("save the current car as a build", self.build.name)

    def _prompt_done(self, value: str) -> None:
        kind = self._prompt_kind
        value = value.strip()
        if not value:
            return
        if kind == "rename" and self.designer is not None:
            self.designer.spec.name = value[:32]
            self.designer.dirty = True
            self.designer.update()
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
            #  'kestrel-fast') is saved beside it instead
            name = value[:32]
            new = name if name in self.lib.builds else self.lib.unique_name("builds", name)
            self.build.name = new
            try:
                self.lib.save_build(self.build.to_json())
            except (OSError, ValueError) as exc:
                self.hint = _could_not_save(exc)
                return
            self.lib_page.refresh()
            self.hint = (f"build '{new}' saved to the library" if new == name
                         else f"'{name}' already exists - saved as '{new}'")

    # -- slot editing (car page) ---------------------------------------------
    def _cycle_wing(self, d: int = 1) -> None:
        slot = self.build.slot(self.sel)
        role = SLOT_ROLE[self.sel]
        names = [""] + sorted(n for n, w in self.lib.wings.items() if w.role == role)
        i = names.index(slot.wing) if slot.wing in names else 0
        slot.wing = names[(i + d) % len(names)]
        self.build.sync_mirror(self.sel)
        self.build.clamp(self.lib)
        if self.designer is not None and self.designer.key == self.sel:
            self.designer = None
        self.hint = f"{self.sel}: {slot.wing or 'none'}"

    def _move(self, dx: float = 0.0, dh: float = 0.0, dinc: float = 0.0) -> None:
        slot = self.build.slot(self.sel)
        slot.x += dx
        slot.h += dh
        slot.inc_deg += dinc
        self.build.clamp(self.lib)
        self.build.sync_mirror(self.sel)
        if self.designer is not None and self.designer.key == self.sel:
            self.designer.update()

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
            if self.page == "section":
                self._run_optimiser()
            elif self.page == "airfoil":
                self.af_page.request_xfoil()
            elif self.page == "library":
                self._save_build_quick()
        if e["triangle"]:
            if self.page == "section":
                self.design_page.focus = ("rows" if self.design_page.focus == "nav" else "nav")
            elif self.page == "library":
                self.lib_page.focus = "builds" if self.lib_page.focus == "wings" else "wings"
            elif self.page == "airfoil":
                self.af_page.focus = "params" if self.af_page.focus == "list" else "list"
        if e["options"]:
            self._menu_open()
        return None

    # -- page-generic navigation ------------------------------------------------
    def _page_nav(self, d: int) -> None:
        if self.page == "mission":
            self.mission_page.params.nav(d)
        elif self.page == "section":
            (self.design_page.nav if self.design_page.focus == "nav"
              else (self.design_page.rank_list if self.design_page.nav.current().endswith(".rank")
                    else self.design_page.rows() or self.design_page.nav)).nav(d)
        elif self.page == "airfoil":
            if self.af_page.focus == "list":
                self.af_page.list.nav(d)
            else:
                self.af_page.params.nav(d)
        elif self.page == "library":
            (self.lib_page.wings if self.lib_page.focus == "wings" else self.lib_page.builds).nav(d)

    def _page_adjust(self, d: int, fine: bool) -> None:
        if self.page == "mission":
            self.mission_page.params.adjust(d, fine)
        elif self.page == "section":
            dp = self.design_page
            if dp.focus == "nav":
                dp.nav.nav_group(d)
            else:
                rows = dp.rows()
                if rows is not None:
                    rows.adjust(d, fine)
        elif self.page == "airfoil" and self.af_page.focus == "params":
            self.af_page.params.adjust(d, fine)
        elif self.page == "airfoil":
            self.af_page.list.nav(d)

    def _page_activate(self) -> None:
        #  On the two new pages ENTER means THE STEP: state the mission, accept
        #  the section. That is what the hint bar promises, and the rows there
        #  are changed with LEFT/RIGHT. Everywhere else ENTER keeps its old
        #  meaning of "activate this row" -- which for a choice row is a cycle,
        #  so without this branch ENTER on the mission page cycled the circuit
        #  and the step never happened.
        if self.page == "mission":
            p = self.mission_page.params.current()
            if p is not None and p.kind == "action":
                p.activate()
            else:
                self.mission_page.state()
        elif self.page == "section":
            self.design_page.act()
            self.hint = self.design_page.msg or self.hint
        elif self.page == "airfoil":
            if self.af_page.focus == "list":
                self.assign_airfoil()
            else:
                self.af_page.params.activate()
        elif self.page == "library":
            self._library_select()

    # -- THE MOUSE ---------------------------------------------------------------
    #  Every page is driven by the keyboard OR the mouse, and the mouse does
    #  the SAME things: it lands on the row the drawing put under the cursor
    #  and does there what the keyboard would do on that row. The navigator's
    #  gate applies to a click exactly as it applies to an arrow key -- the
    #  UROP app's own shell refuses a locked node's click and notifies the
    #  reason, which is what `Nav.select` does here.
    def _page_widgets(self):
        """(widget, focus-name) pairs a click may land on, on THIS page."""
        if self.page == "mission":
            return [(self.mission_page.params, "rows")]
        if self.page == "section":
            dp = self.design_page
            out = [(dp.nav, "nav")]
            if dp.nav.current().endswith(".rank"):
                out.append((dp.rank_list, "rows"))
            rows = dp.rows()
            if rows is not None:
                out.append((rows, "rows"))
            return out
        if self.page == "airfoil":
            return [(self.af_page.list, "list"), (self.af_page.params, "params")]
        if self.page == "library":
            return [(self.lib_page.wings, "wings"), (self.lib_page.builds, "builds")]
        return []

    def _page_click(self, pos, fine: bool) -> None:
        for w, focus in self._page_widgets():
            if isinstance(w, ui.Nav):
                if not w.hit(pos):
                    continue
                self.design_page.focus = "nav"
                if w.click(pos):
                    self.design_page.refresh_rank()
                self.hint = w.refused or self.hint
                return
            if w.hit(pos) < 0:
                continue
            self._set_focus(focus)
            if isinstance(w, ui.ListBox):
                if w.click(pos) == "activate":
                    self._page_activate()
                elif self.page == "section":
                    self.design_page.act()
                    self.hint = self.design_page.msg or self.hint
                return
            r = w.click(pos, fine)
            if r in ("action", "adjust"):
                self._after_rows_changed()
            return

    def _page_wheel(self, dy: int) -> None:
        pos = pygame.mouse.get_pos()
        for w, focus in self._page_widgets():
            hit = bool(w.hit(pos)) if isinstance(w, ui.Nav) else (w.hit(pos) >= 0)
            if not hit:
                continue
            self._set_focus(focus)
            if isinstance(w, ui.Nav):
                w.nav(1 if dy < 0 else -1)
                self.design_page.refresh_rank()
            else:
                w.wheel(dy)
            return

    def _set_focus(self, focus: str) -> None:
        if self.page == "section":
            self.design_page.focus = focus
        elif self.page == "airfoil":
            self.af_page.focus = focus
        elif self.page == "library":
            self.lib_page.focus = focus

    def _after_rows_changed(self) -> None:
        """A row fired or stepped: the gates may have moved with it."""
        if self.page == "section":
            self.design_page.settle()
            dp = self.design_page
            self.hint = (dp.msg or dp.af.msg or dp.ep.msg
                         or (dp.wing.msg if dp.wing else "") or self.hint)

    def _run_optimiser(self, extend: bool = False) -> None:
        """The SECTION page and the WING page share this: both are a GP-BO run
        against the same stated mission, and both want the 'optimising...'
        frame on screen before the run blocks the loop. `extend` continues
        the wing's last run (AeroBO's "keep going")."""
        page = self.design_page if self.page == "section" else self.designer
        if page is None:
            return
        self.hint = "optimising..." if not extend else "keep going..."
        self._draw_page()
        pygame.display.flip()
        page.optimise(extend=extend)
        self.hint = page.msg or "optimiser done"

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
            slot = self.build.slot(self.sel)
            slot.wing = w.name
            self.build.sync_mirror(self.sel)
            self.build.clamp(self.lib)
            self.designer = None
            self.hint = f"{self.sel}: {w.name}"
            self.page = "car"
        else:
            it = lp.builds.current()
            if not it:
                return
            self.build = CarBuild.from_json(self.lib.builds[it[0]]).clamp(self.lib)
            self.designer = None
            self.hint = f"loaded build '{self.build.name}'"
            self.page = "car"

    def _save_build_quick(self) -> None:
        name = self.lib.unique_name("builds", self.build.name)     # never over another build
        self.build.name = name
        try:
            self.lib.save_build(self.build.to_json())
        except (OSError, ValueError) as exc:
            self.hint = _could_not_save(exc)
            return
        self.lib_page.refresh()
        self.hint = f"build '{name}' saved"

    def _new_wing(self) -> None:
        slot = self.build.slot(self.sel)
        slot.wing = ""
        self.build.sync_mirror(self.sel)
        self.designer = None
        self.section_for.pop(self.sel, None)     # a new wing re-walks the chain
        self.open_mission()

    def _delete_library_item(self) -> None:
        lp = self.lib_page
        if lp.focus == "wings":
            it = lp.wings.current()
            if it and it[0] in self.lib.wings:
                if self.lib.wings[it[0]].builtin:
                    self.hint = "built-in wings cannot be deleted"
                    return
                self.lib.delete("wings", it[0])
                self.build.clamp(self.lib)
                self.hint = f"deleted wing '{it[0]}'"
        else:
            it = lp.builds.current()
            if it and it[0] in self.lib.builds:
                self.lib.delete("builds", it[0])
                self.hint = f"deleted build '{it[0]}'"
        lp.refresh()

    # -- keyboard / mouse -----------------------------------------------------
    def _handle(self, ev) -> str | None:
        if ev.type == pygame.QUIT:
            return "quit"
        if self.prompt.open:
            r = self.prompt.handle(ev)
            if r == "ok":
                self._prompt_done(self.prompt.value)
            return None
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
            if ev.key == pygame.K_h and self.tutor is not None and self.tutor.active:
                self.tutor.hidden = not self.tutor.hidden
                return None
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
            self.build.reset()
            self.build.clamp(self.lib)
            self.designer = None
            self.hint = "car reset: no wings"
        elif k == pygame.K_c:
            self.cam.reset()
        elif k == pygame.K_v:
            self.view.show_vectors = not self.view.show_vectors
            self.hint = "force vectors " + ("shown" if self.view.show_vectors else "hidden")
        elif k == pygame.K_d:
            self.open_mission()
        elif k == pygame.K_a:
            self.open_airfoils()
        elif k == pygame.K_l:
            self.open_library()
        return None

    def _handle_page_key(self, ev) -> str | None:
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
            if self.page == "section":
                self.design_page.focus = ("rows" if self.design_page.focus == "nav" else "nav")
            elif self.page == "library":
                self.lib_page.focus = "builds" if self.lib_page.focus == "wings" else "wings"
            elif self.page == "airfoil":
                self.af_page.focus = "params" if self.af_page.focus == "list" else "list"
        elif self.page == "section":
            dp = self.design_page
            if k == pygame.K_o:
                self._run_optimiser()
            elif k == pygame.K_k:
                self._run_optimiser(extend=True)
            elif k == pygame.K_l:
                self.hint = "screening the library..."
                self._draw_page()
                pygame.display.flip()
                dp.screen()
                self.hint = dp.msg
            elif k == pygame.K_f:
                dp.fit_current()
                self.hint = dp.msg or dp.nav.refused
            elif k == pygame.K_s:
                self.designer = dp.wing
                dp.wing.commit()
                self.hint = dp.wing.msg
            elif k == pygame.K_x:
                dp.wing.request_xfoil()
                self.hint = dp.wing.msg
            elif k == pygame.K_n:
                self.designer = dp.wing
                self.prompt_rename()
            elif k == pygame.K_a:
                self.designer = dp.wing
                self.open_airfoils()
        elif self.page == "airfoil":
            if k == pygame.K_x:
                self.af_page.request_xfoil()
            elif k == pygame.K_n:
                self.prompt_naca()
        elif self.page == "library":
            if k == pygame.K_s:
                self.prompt_build()
            elif k in (pygame.K_DELETE, pygame.K_BACKSPACE):
                self._delete_library_item()
            elif k == pygame.K_n:
                self._new_wing()
        return None

    # -- drawing ------------------------------------------------------------------
    def _draw_page(self) -> None:
        u = self.view.ui
        pad_name = self.pad.name if self.pad is not None else None
        if self.page == "car":
            dep = self.deploy * self.deploy * (3.0 - 2.0 * self.deploy)
            self.view.draw(self.build, self.lib, self.cam, dep, self.sel, pad_name, self.hint, self.status)
            return
        self.screen.fill(C_BG)
        help_ = ""
        if self.page == "mission":
            help_ = self.mission_page.draw(self.screen, self.text, self.plot, u)
            hints = [("CLICK", "a row, or a value's < >"), ("UP/DOWN", "row"),
                     ("LEFT/RIGHT", "circuit / surface"),
                     ("ENTER", "state it -> design"), ("ESC", "back")]
            pad_h = [("stick/d-pad", "row / adjust"), ("CROSS", "state it"),
                     ("CIRCLE", "back")] if pad_name else None
        elif self.page == "section":
            help_ = self.design_page.draw(self.screen, self.text, self.plot, u)
            hints = [("CLICK", "a step, a row, a value's < >"),
                     ("TAB", "steps / rows"), ("UP/DOWN", "move"),
                     ("LEFT/RIGHT", "adjust (SHIFT fine)"), ("ENTER", "do this step"),
                     ("L", "screen"), ("O", "optimise"), ("F", "fit to the wing"),
                     ("S", "save"), ("ESC", "back")]
            pad_h = [("stick/d-pad", "move / adjust"), ("TRIANGLE", "steps / rows"),
                     ("CROSS", "do this step"), ("SQUARE", "optimise"),
                     ("CIRCLE", "back")] if pad_name else None
        elif self.page == "airfoil":
            help_ = self.af_page.draw(self.screen, self.text, self.plot, u)
            hints = [("CLICK", "a section (twice to use it)"), ("UP/DOWN", "section"),
                     ("TAB", "list / weights"), ("ENTER", "use this section"),
                     ("X", "XFOIL polar"), ("N", "new NACA"), ("ESC", "back")]
            pad_h = [("stick", "section"), ("CROSS", "use"), ("SQUARE", "XFOIL"), ("TRIANGLE", "list/weights"),
                     ("CIRCLE", "back")] if pad_name else None
        else:
            self.lib_page.draw(self.screen, self.text, u)
            hints = [("UP/DOWN", "item"), ("TAB", "wings / builds"), ("ENTER", "use / load"),
                     ("S", "save car as build"), ("N", "new wing"), ("DEL", "delete"), ("ESC", "back")]
            pad_h = [("stick", "item"), ("TRIANGLE", "wings/builds"), ("CROSS", "use / load"),
                     ("SQUARE", "save build"), ("CIRCLE", "back")] if pad_name else None
        title = {"mission": "MISSION", "section": "DESIGN",
                 "airfoil": "AIRFOIL LIBRARY",
                 "library": "WING & BUILD LIBRARY"}[self.page]
        if self.page == "mission":
            title = "MISSION  [1/2 mission > design]"
        elif self.page == "section":
            dp = self.design_page
            title = (f"DESIGN  [2/2]   {dp.nav.group_of(dp.nav.current()).split('  ')[0]}"
                     f" > {dp.nav.label(dp.nav.current())}")
        r = ui.key_hint_bar(self.screen, self.text, (int(12 * u), int(690 * u), int(1256 * u), int(98 * u)),
                            hints, pad_h, title=f"{title}   {self.build.summary(self.lib)[:100]}")
        if help_:
            self.text.blit(self.screen, help_[:150], r.x + 10, r.bottom - 20, 12, ui.C_KEY)
        if self.status:
            self.text.blit(self.screen, self.status, r.right - 10, r.y + 6, 12, C_TEXT_DIM, right=True)

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
        if action == "drive" and self._t_alive < 0.35:
            action = None                  # a key / button carried in from the drive
        if action == "drive" and self.page != "car":
            self.close_page()
        if self.tutor is not None:
            self.tutor.update(self, action)
        # deploy preview: the same 0.45 s / 0.30 s actuator as vehicle.py
        if self.deploy_cmd > self.deploy:
            self.deploy = min(1.0, self.deploy + dt / 0.45)
        elif self.deploy_cmd < self.deploy:
            self.deploy = max(0.0, self.deploy - dt / 0.30)
        # XFOIL results arriving from the worker
        for name, reb, okp in self.lib.poll():
            if okp:
                for w in self.lib.wings.values():
                    if w.airfoil == name and not w.legacy:
                        ride = w.aero.get("ride_h")
                        self.lib.analyse_wing(w, ride_h=ride)
                        try:
                            self.lib.save_wing(w)
                        except (OSError, ValueError) as exc:
                            self.hint = _could_not_save(exc)
                if self.designer is not None and self.designer.spec.airfoil == name:
                    self.designer.update()
                self.af_page.on_polar(name)
                self.lib_page.refresh()
                self.hint = f"XFOIL polar for {name} arrived (Re {reb:.2g})"
            else:
                self.hint = f"XFOIL did not converge for {name}: estimate kept"
        busy = self.lib.xfoil_busy
        self.status = (f"XFOIL: {busy} ({self.lib.xfoil_pending} queued)" if busy else
                       ("XFOIL available" if self.lib.use_xfoil else "XFOIL not found: estimate polars"))
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
                    self.build.clamp(self.lib)
                    return action
        finally:
            pygame.key.set_repeat()


# =========================================================================== #
#  SELF-CHECK                                                                  #
# =========================================================================== #
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
    physics paths, every page headless, the optimiser, the library."""
    import shutil
    import tempfile
    ok = True

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and passed
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
    key(pygame.K_v)
    rep("V hides the vectors", not g.view.show_vectors, g.hint)
    key(pygame.K_v)

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
    rep("ENTER -> drive", key(pygame.K_RETURN) == "drive", "")

    # --- the designer
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
    rep("the mission lap flies and is quoted against the bare car",
        mp.result is not None and mp.result.ok and mp.bare is not None and mp.bare.ok,
        f"{mp.result.time:.3f} s vs {mp.bare.time:.3f} s bare, "
        f"{len(mp.profile().corners)} corners")

    #  the GATE, which is the point of the mission being first
    g.open_designer("left")
    rep("the design page is GATED on a stated mission", g.page == "mission",
        g.hint)

    #  STEP 2: the DESIGN navigator.
    key(pygame.K_RETURN)
    dp = g.design_page
    rep("ENTER states the mission and opens the DESIGN navigator",
        g.page == "section" and g.mission.stated and dp.wing is not None,
        f"{g.page}, at '{dp.nav.current()}'")
    rep("the navigator carries the whole procedure, in order",
        [k for _, st in DESIGN_TREE for k, _ in st] == dp.nav.keys and len(dp.nav.keys) == 16,
        " > ".join(gname.split("  ")[0] for gname, _ in DESIGN_TREE))
    #  every one of the sixteen panels draws, which is the cheapest way to
    #  catch a panel that reaches for a key its model has not filled in yet.
    worst, worst_k = 0.0, ""
    for node in dp.nav.keys:
        #  FORCED: the gate refuses most of these on a page nobody has
        #  worked yet, and a panel that never drew is a panel that was never
        #  checked. The gate itself is tested below, on the real selection.
        dp.nav.select(node, force=True)
        t0 = time.perf_counter()
        g._draw_page()
        dt = (time.perf_counter() - t0) * 1e3
        if dt > worst:
            worst, worst_k = dt, node
    rep("all 16 navigator panels draw", True, f"worst {worst:.1f} ms at '{worst_k}'")

    #  THE GATE. Nothing downstream may be selected before the step that
    #  feeds it is finished -- AeroBO's own `select()` refuses a locked node
    #  and notifies the stored reason, and this is that, one page down.
    dp.open("left")
    rep("a fresh page opens on the FIRST step and everything else is shut",
        dp.nav.current() == "af.screen"
        and sum(dp.nav.open(k) for k in dp.nav.keys) == 1,
        f"{sum(dp.nav.open(k) for k in dp.nav.keys)} of {len(dp.nav.keys)} open")
    rep("jumping to the wing is REFUSED, with the reason",
        dp.nav.select("w.box") is False and dp.nav.current() == "af.screen"
        and "AIRFOIL" in dp.nav.refused,
        dp.nav.refused[:64])
    rep("...and so is a click on it",
        dp.nav.click((0, 0)) == "" and dp.nav.current() == "af.screen")
    #  each gate opens with ONE action
    dp.af.screen_library()
    rep("screening opens the ranking and the section",
        all(dp.nav.open(k) for k in ("af.rank", "af.section", "af.opt"))
        and not dp.nav.open("ep.screen"),
        "af.* open, ep.* still shut")
    #  A COMPOSITE WINNER MUST NOT MOVE THE MOUNT ANGLE. Its incidence
    #  coordinate is inert -- the lattice is never built -- so carrying it
    #  across would be a design decision taken by a number nothing scored.
    dp.af.objective = "composite"
    dp.af.rebuild()
    inc_was = g.build.left.inc_deg
    dp.af.x = np.asarray(dp.af.x, dtype=float).copy()
    dp.af.x[-1] = BOUNDS["flank"]["inc_deg"][0]        # park it at the band edge
    dp.af.evaluate()
    dp.fit(dp.af)
    rep("a COMPOSITE winner does not carry its incidence to the wing",
        abs(g.build.left.inc_deg - inc_was) < 1e-9,
        f"row parked at {dp.af.x[-1]:+.1f} deg, mount still {g.build.left.inc_deg:+.2f}")
    dp.af.objective = "lap time"
    dp.af.rebuild()
    dp.af.evaluate()
    dp.fit(dp.af)
    rep("...but a LAP winner does, because the lap flew it",
        abs(g.build.left.inc_deg - float(dp.af.res["inc_deg"])) < 1e-6,
        f"mount {g.build.left.inc_deg:+.2f} deg")

    n_before = len(g.lib.airfoils)
    dp.fit(dp.af)
    rep("fitting the section to the wing finishes the AIRFOIL group",
        dp.af.finished() and dp.nav.open("ep.screen")
        and len(g.lib.airfoils) == n_before + 1,
        "ENDPLATE opens")
    rep("...but the WING is still shut on the plates' question",
        not dp.nav.open("w.box") and "flat" in dp._reason("w.box"),
        dp._reason("w.box")[:62])
    dp.ep.decline()
    rep("declining the plates is an ANSWER and opens the wing",
        dp.nav.open("w.box") and dp.nav.open("w.type"),
        dp.ep.msg[:60])
    #  ...and a plate at zero height is LOCKED, not blocked: no work upstream
    #  opens it, which is a different sentence and a different mark
    h_was = dp.wing.spec.plate_h
    dp.wing.spec.plate_h = 0.0
    rep("a plate at zero height is LOCKED, not blocked",
        dp._state("ep.section") == "locked"
        and "zero height" in dp._reason("ep.section"),
        dp._reason("ep.section")[:62])
    dp.wing.spec.plate_h = h_was
    dp.wing.update()

    #  the MOUSE. Every page takes it, and it does what the keyboard does.
    dp.nav.select("af.screen", force=True)
    g._draw_page()                                    # records the geometry
    hit_key = dp.nav._hits[2][0] if len(dp.nav._hits) > 2 else ""
    yy = dp.nav._hits[2][1] + 2 if len(dp.nav._hits) > 2 else 0
    took = dp.nav.click((dp.nav._rect.x + 60, yy))
    rep("a click on the navigator selects that step",
        took == hit_key and dp.nav.current() == hit_key, f"clicked '{hit_key}'")
    dp.nav.select("af.screen", force=True)
    dp.focus = "rows"
    g._draw_page()
    rows = dp.rows()
    rows.select_key("w.clmax")
    g._draw_page()
    i = [j for j, pm in enumerate(rows.params) if pm.key == "w.clmax"][0]
    yrow = [ry for jj, ry, _ in rows._hits if jj == i][0]
    w0 = dp.af.weights["clmax"]
    rows.click((rows._rect.right - 30, yrow + 4))     # the right-hand arrow
    rep("a click on the right of a value row steps it up",
        dp.af.weights["clmax"] > w0,
        f"clmax {w0:.2f} -> {dp.af.weights['clmax']:.2f}")
    rows.click((rows._rect.right - 130, yrow + 4))    # the left-hand arrow
    rep("...and on the left, down", abs(dp.af.weights["clmax"] - w0) < 1e-12,
        f"back to {dp.af.weights['clmax']:.2f}")
    rows.select_key("go")
    g._draw_page()
    iy = [ry for jj, ry, _ in rows._hits
          if rows.params[jj].key == "go"][0]
    dp.af.ranked = []
    rows.click((rows._rect.x + 40, iy + 4))           # already selected: fires
    rep("a click on an action row runs it",
        len(dp.af.ranked) == len(g.lib.airfoils),
        f"{len(dp.af.ranked)} sections screened by the mouse")
    dp.focus = "nav"
    #  the gate and the mouse were exercised on a page that is now half
    #  worked and carries edited weights. Everything after this walks the
    #  chain for real, so it starts from a fresh one.
    dp.open("left")
    #  ...and the WEIGHTS are deliberately NOT reset by `open`: editing one
    #  makes the set the user's and nothing moves it again, which is AeroBO's
    #  rule. The mouse check above edited one, so the check puts it back.
    dp.af._set_preset(scr.recommended("flank", "main"))
    dp.nav.select("af.screen", force=True)
    t0 = time.perf_counter()
    g.frame(1.0 / 60.0)
    g.frame(1.0 / 60.0)
    mspf = (time.perf_counter() - t0) * 1e3 / 2
    rep("design page draws", *_budget(mspf, 90.0))
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_section.png"))

    #  AIRFOIL: screen -> rank -> section -> optimise. THE WEIGHTS ARE THE
    #  SCREEN'S QUESTION -- AeroBO's order, and the one thing this page was
    #  asking in the wrong step.
    dp.nav.select("af.screen")
    keys = [pm.key for pm in dp.rows().params]
    rep("the criterion weights are on the SCREENING step",
        all(f"w.{c}" in keys for c in scr.CRITERIA)
        and "preset" in keys and "tcmin" in keys and "fclmax" in keys,
        f"{len(scr.CRITERIA)} weights, a preset, two gates and three floors")
    rep("...and each form opens on the preset measured for ITS surface",
        dp.af.preset == "wing (downforce)"
        and dp.ep.preset == scr.recommended(SLOT_ROLE[dp.key], "plate"),
        f"airfoil '{dp.af.preset}', endplate '{dp.ep.preset}' "
        f"(a {SLOT_ROLE[dp.key]} plate)")
    rep("the default wing weights do NOT rank a car wing on |cm|",
        dp.af.weights["cm"] == 0.0,
        "|cm| is lower-better, so it rewards REFLEX -- and it ranks the "
        "library backwards here (rho -0.97 against the lap)")
    rep("a plate is not ranked on the two criteria it cannot carry",
        sorted(dp.ep.dead()) == ["ldcr", "ldmax"] and dp.ep.dead_weighted() == [],
        "both L/D criteria are read at a lift a vertical panel never carries; "
        "|cm| is NOT retired here, because carsim's plate may be cambered")
    #  the OPTIMISER's defaults are AeroBO's measured law, not a guess
    rep("the section budget is the measured law at this problem's row count",
        dp.af.budget == opt.budget_for(10) == 40
        and dp.ep.budget == opt.budget_for(9) == 37,
        f"section {dp.af.budget} evals (d=10), plate {dp.ep.budget} (d=9)")
    rep("...and the wing's is the same law at the wing's row count",
        dp.wing.budget == opt.budget_for(len(design_vars(dp.wing.role))) == 31,
        f"{dp.wing.budget} evals (d={len(design_vars(dp.wing.role))})")
    rep("the Sobol start is 0.5 x d, not half the budget",
        opt.split_for(10, dp.af.budget) == (5, 35),
        f"{opt.split_for(10, dp.af.budget)[0]} of {dp.af.budget}, where it "
        f"used to be {min(16, 48 // 2)} of 48")
    dp.af._set_effort("thorough")
    rep("effort re-sizes the budget through the law",
        dp.af.budget == opt.budget_for(10, "thorough") > 40,
        f"thorough -> {dp.af.budget} evals")
    dp.af._set_effort("balanced")
    #  a composite search cannot start before the band has been measured
    dp.af.objective = "composite"
    rep("a composite search is BLOCKED until the library has been screened",
        dp.af.state("opt") == "blocked" and dp.af.optimise() is None,
        dp.af.msg[:70])

    t0 = time.perf_counter()
    dp.af.screen_library()
    dt = time.perf_counter() - t0
    n_seen = len(dp.af.ranked) + len(dp.af.refused)
    rep("the library screen scores every section on the WEIGHTS",
        n_seen == len(g.lib.airfoils) and dt < 20.0 and bool(dp.af.band),
        f"{len(dp.af.ranked)} ranked, {len(dp.af.refused)} gated, in {dt:.1f} s, "
        f"best '{dp.af.seed}' ({len(dp.af.clipped)} clipped into the box)")
    rep("and it is ranked best-first on the composite",
        all(a[1] >= b[1] for a, b in zip(dp.af.ranked, dp.af.ranked[1:])
            if math.isfinite(a[1]) and math.isfinite(b[1])))
    rep("the band is frozen from that screen, so the score is a fixed map",
        set(dp.af.band) and dp.af.prob.band is dp.af.band,
        ", ".join(sorted(dp.af.band)))
    rep("the winner's own sub-scores became the seed the goal is held to",
        dp.af.seed_sub is not None and set(dp.af.seed_sub) == set(dp.af.band))

    #  THE WEIGHTS DECIDE THE SHORTLIST. Rank once on the shipped preset and
    #  once on thickness alone: the second must return the thickest section.
    was, best0 = dict(dp.af.weights), dp.af.ranked[0][0]
    dp.af.weights.clear(); dp.af.weights["thick"] = 1.0
    dp.af.screen_library()
    thickest = max(dp.af.ranked, key=lambda r: r[2]["metrics"]["thick"])[0]
    rep("a weight edit re-chooses the shortlist",
        dp.af.ranked[0][0] == thickest,
        f"preset -> '{best0}',  thickness alone -> '{dp.af.ranked[0][0]}'")
    #  ...and a GATE drops candidates rather than ranking them low
    dp.af.weights.clear(); dp.af.weights.update(was)
    dp.af.gates["tc_min"] = 0.12
    dp.af.screen_library()
    rep("a gate DROPS candidates instead of ranking them low",
        len(dp.af.refused) > 0
        and all(r[2]["metrics"]["thick"] >= 0.12 - 1e-9 for r in dp.af.ranked),
        f"{len(dp.af.refused)} of {len(g.lib.airfoils)} refused at t/c >= 0.12")
    dp.af.gates.update(scr.GATES_OFF_DEFAULT)
    dp.af.objective = "lap time"
    dp.af.screen_library()

    #  the ranking table says WHERE a score came from, not only what it was
    dp.nav.select("af.rank")
    dp.refresh_rank()
    row0 = dp.rank_list.items[0][1]
    rep("the ranking table carries every criterion and its points",
        all(h in row0 for _, h, _ in dp.RANK_COLUMNS) and "(" in row0,
        row0[:92])

    rep("the winner is loaded into the SHAPE",
        dp.af.x is not None and len(dp.af.x) == 2 * sec.N_CST + 2,
        f"{len(dp.af.x)} rows")
    #  the CST shape weights are EDITABLE and moving one moves the section
    dp.nav.select("af.section")
    dp.focus = "rows"
    dp.af.params.select_key("wu1")
    tc0, cam0 = dp.af.res["geometry"]["tc"], dp.af.res["geometry"]["camber"]
    key(pygame.K_RIGHT)
    rep("RIGHT on a weight row reshapes the section and re-flies it",
        abs(dp.af.res["geometry"]["camber"] - cam0) > 1e-6,
        f"camber {cam0:+.4f} -> {dp.af.res['geometry']['camber']:+.4f}")
    dp.af.params.select_key("tc")
    key(pygame.K_RIGHT)
    rep("and the t/c row lands exactly where it says",
        abs(dp.af.res["geometry"]["tc"] - (tc0 + 0.005)) < 1e-6,
        f"t/c {tc0:.4f} -> {dp.af.res['geometry']['tc']:.4f}")
    dp.af.budget = 16
    t0 = time.perf_counter()
    run = dp.af.optimise()
    dt = time.perf_counter() - t0
    rep("the section optimiser runs and reports BO vs random",
        run is not None and run["n"] == 16 and dt < 40.0 and math.isfinite(run["bo"]),
        f"BO {dp.af.fmt_score(run['bo'])} random {dp.af.fmt_score(run['rs'])} "
        f"start {dp.af.fmt_score(run['start'])} in {dt:.1f} s")
    n_af = len(g.lib.airfoils)
    dp.fit(dp.af)
    rep("F saves the designed section and fits it to the wing",
        len(g.lib.airfoils) == n_af + 1 and dp.wing.spec.airfoil in g.lib.airfoils,
        f"section '{dp.wing.spec.airfoil}'")
    rep("the wing inherited its incidence, inside the wing's own band",
        BOUNDS["flank"]["inc_deg"][0] - 1e-9 <= g.build.left.inc_deg
        <= BOUNDS["flank"]["inc_deg"][1] + 1e-9, f"inc {g.build.left.inc_deg:+.2f} deg")

    #  ENDPLATE: the same four stages, pointed at the tip panels
    dp.wing.spec.plate_h = max(dp.wing.spec.plate_h, 0.14)
    dp.wing.update()
    dp.ep.rebuild()
    a_flat = dict(dp.wing.spec.aero)
    rep("a plate's design vector has NO incidence row",
        dp.ep.prob.bounds().shape[0] == 2 * sec.N_CST + 1,
        f"{dp.ep.prob.bounds().shape[0]} rows against the airfoil's {2 * sec.N_CST + 2}")
    dp.ep.screen_library()
    rep("the end plate screens the library too", len(dp.ep.ranked) == len(g.lib.airfoils),
        f"{len(dp.ep.ranked)} sections, best '{dp.ep.seed}'")
    dp.ep.seed_from("e423")
    dp.fit(dp.ep)
    a_camb = dict(dp.wing.spec.aero)
    rep("a designed END PLATE reaches the lattice and moves the wing",
        dp.wing.spec.plate_airfoil in g.lib.airfoils
        and abs(a_camb["CL0"] - a_flat["CL0"]) > 1e-3,
        f"plate_h {dp.wing.spec.plate_h:.2f} m: CL0 {a_flat['CL0']:.4f} -> {a_camb['CL0']:.4f} "
        f"({100 * (a_camb['CL0'] / a_flat['CL0'] - 1):+.1f} %), "
        f"CL_max {a_flat['CL_max']:.4f} -> {a_camb['CL_max']:.4f}")

    #  ...and HOW THE TWO MEET, which is a WING TYPE choice: it decides what
    #  geometry is being built, not where inside it to search
    by_key = {p.key: p for p in dp.wing.params.params}
    dp.nav.select("w.type", force=True)
    type_keys = [r.key for r in dp.rows().params if r.kind != "label"]
    rep("the transition is asked on WING TYPE, beside the mount",
        type_keys[:4] == ["mount", "blend", "bshape", "junc"],
        "  ".join(type_keys))
    a_sq = dict(dp.wing.spec.aero)
    by_key["blend"].adjust(+1)            # 0.00 -> 0.05, the same as RIGHT
    rep("raising the blend off zero switches the JUNCTION CHARGE on with it",
        dp.wing.spec.plate_blend > 0.0 and dp.wing.spec.plate_junction,
        "AeroBO's own behaviour: the credit for softening a corner IS the "
        "reason to soften it, and the lattice cannot see a corner")
    for _ in range(11):
        by_key["blend"].adjust(+1)
    by_key["bshape"].set("spiral")
    a_bl = dict(dp.wing.spec.aero)
    rep("...and the wing is re-flown on the blended geometry",
        abs(a_bl["plate_projection"]) > 1e-4 and a_bl["e"] > a_sq["e"]
        and a_bl["span_flown"] < a_sq["span_flown"]
        and 0.0 < a_bl["cd_junction"] < a_sq.get("cd_junction", 1.0) + 1.0,
        f"blend {dp.wing.spec.plate_blend:.2f} spiral: reach "
        f"{a_bl['plate_projection']:.4f} m a side, span {a_sq['span_flown']:.3f} -> "
        f"{a_bl['span_flown']:.3f} m, e {a_sq['e']:.3f} -> {a_bl['e']:.3f}, "
        f"junction {a_bl['cd_junction'] * 1e4:.1f} ct")
    dp.wing.spec.plate_blend = 0.0
    dp.wing.spec.plate_junction = False
    dp.wing.spec.plate_shape = "arc"
    dp.wing.dirty = True
    dp.wing.update()
    rep("dropping the blend back to zero restores the square corner exactly",
        all(dp.wing.spec.aero[k] == a_camb[k] for k in ("CL0", "CLa", "cd0", "e", "S")),
        "a setting, and reversible like one")
    ph = dp.wing.spec.plate_h
    dp.wing.spec.plate_h, dp.wing.spec.mount = 0.0, "pylon"
    rep("no plate, nothing to blend: the rows say so rather than taking an edit",
        not by_key["blend"].enabled and not by_key["bshape"].enabled
        and not by_key["junc"].enabled and not by_key["blend"].adjust(+1),
        "the same lock the ENDPLATE group carries")
    dp.wing.spec.plate_h, dp.wing.spec.mount = ph, "endplate"
    dp.wing.dirty = True
    dp.wing.update()
    dp.nav.select("w.box", force=True)

    #  WING: the design box, the solver, the convergence
    dsn = dp.wing
    g.designer = dsn
    dp.nav.select("w.box")
    dp.focus = "rows"
    rows = dp.rows()
    rows.select_key("span")
    s0 = dsn.spec.span
    key(pygame.K_RIGHT)
    rep("RIGHT on the span row widens the wing and re-analyses",
        abs(dsn.spec.span - s0 - 0.02) < 1e-9 and dsn.dirty and "CLa" in dsn.spec.aero,
        f"span {s0:.3f} -> {dsn.spec.span:.3f}, CLa {dsn.spec.aero.get('CLa', 0):.3f}")
    rows.select_key("inc")
    #  step DOWN from wherever the fitted section left it: the row is clamped
    #  at the band's ceiling, and a seed that lands on it would make this
    #  check about the clamp rather than about the mirror.
    key(pygame.K_LEFT)
    i0 = g.build.left.inc_deg
    key(pygame.K_RIGHT)
    rep("incidence row edits the slot (mirrored)",
        abs(g.build.left.inc_deg - i0 - 0.5) < 1e-9
        and abs(g.build.right.inc_deg - g.build.left.inc_deg) < 1e-9,
        f"inc {g.build.left.inc_deg:+.1f}")

    #  THE DESIGN BOX IS A BAND TABLE, which is what AeroBO calls a design box
    #  and what this page used to leave as module constants nobody could
    #  reach. A band row must bind the SEARCH, not just print.
    rows.select_key("bx.taper.max")
    rep("every design variable carries a min and a max row",
        all(f"bx.{a}.{t}" in [pm.key for pm in rows.params]
            for a in design_vars(dsn.role) for t in ("min", "max")),
        f"{2 * len(design_vars(dsn.role))} band rows")
    dsn.box["taper"][1] = 0.50
    b_nar, _lab = dsn.search_bounds()
    i_taper = design_vars(dsn.role).index("taper")
    rep("narrowing a band narrows the box the search is given",
        abs(b_nar[i_taper][1] - 0.50) < 1e-12,
        f"taper band {b_nar[i_taper][0]:.2f} .. {b_nar[i_taper][1]:.2f}")
    #  ...and it may not be inverted, nor opened past what the car can take
    dsn._set_box("taper", 0)(9.9)
    rep("a band cannot be inverted or opened past the packaging band",
        dsn.box["taper"][0] < dsn.box["taper"][1]
        and dsn.box["taper"][0] <= BOUNDS[dsn.role]["taper"][1],
        f"taper {dsn.box['taper'][0]:.3f} .. {dsn.box['taper'][1]:.3f}")
    dsn.box["taper"] = list(BOUNDS[dsn.role]["taper"])
    #  the SPAN's upper end is still the car's fit, whatever is typed
    #  (task 41: the fit is the owner's ground rule, `bodies.span_ceiling`,
    #  so a typed ceiling has to be past it for the cap to show)
    typed = dsn._span_band()[1] + 0.5
    dsn.box["span"][1] = typed
    b_sp, _ = dsn.search_bounds()
    rep("the span band is still capped by the car's span fit",
        abs(b_sp[design_vars(dsn.role).index("span")][1]
            - dsn._span_band()[1]) < 1e-12,
        f"typed {typed:.2f} m, searched "
        f"{b_sp[design_vars(dsn.role).index('span')][1]:.3f} m")
    dsn.box["span"][1] = BOUNDS[dsn.role]["span"][1]

    #  FREEING THE REFERENCE AREA adds a row, one ahead of the span, exactly
    #  where AeroBO's `CarWingProblem` puts it. The whole vector has to move
    #  with it -- the bounds, the start, the labels and the decode.
    n0 = len(design_x0(dsn.spec, g.build.left.inc_deg))
    dsn._set_area_free("searched")
    dp._box_rows = None
    b_free, lab_free = dsn.search_bounds()
    x0_free = design_x0(dsn.spec, g.build.left.inc_deg, area=True)
    rep("freeing the reference area adds the row AeroBO puts ahead of the span",
        len(b_free) == n0 + 1 and len(x0_free) == n0 + 1
        and lab_free[lab_free.index("span (vertical)") - 1] == "reference area",
        f"{n0} rows -> {len(b_free)}: ..., {lab_free[-2]}, {lab_free[-1]}")
    sp_probe = dsn.spec.copy()
    inc_probe = apply_design(sp_probe, x0_free, area=True)
    rep("...and the decode round-trips it",
        abs(sp_probe.area - (dsn.spec.area if dsn.spec.area > 0 else dsn.spec.S)) < 1e-9
        and abs(inc_probe - g.build.left.inc_deg) < 1e-9,
        f"area {sp_probe.area:.4f} m2")
    dsn._set_area_free("fixed")
    dp._box_rows = None

    dsn.budget = 16
    t0 = time.perf_counter()
    res = dsn.optimise()
    dt = time.perf_counter() - t0
    rep("the wing optimiser runs and reports BO vs random",
        res is not None and res["n"] == 16 and dt < 30.0 and math.isfinite(res["bo"]),
        f"BO {res['bo']:.3f} random {res['rs']:.3f} start {res['start']:.3f} in {dt:.1f} s")
    #  AeroBO's "keep going": the 16 are inherited as the GP's training set,
    #  4 more are bought, the trace carries on and the best cannot get worse
    dsn.more = 4
    bo_16 = res["bo"]
    res2 = dsn.optimise(extend=True)
    rep("keep going continues the run: 16 inherited, 4 bought, nothing re-flown",
        res2 is not None and res2["continued"] and res2["n"] == 20 and res2["n_prior"] == 16
        and len(res2["trace"]) == 20 and res2["trace"][:16] == res["trace"][:16]
        and res2["bo"] >= bo_16 - 1e-12,
        f"BO {bo_16:.4f} -> {res2['bo']:.4f} over {res2['n_prior']} -> {res2['n']} evals")
    #  ...and a box that moved under the record is a NEW search, not a
    #  continuation of one that no longer exists
    dsn.box["taper"][1] = 0.60
    dsn.budget = 8
    res3 = dsn.optimise(extend=True)
    rep("keep going on a moved box starts over",
        res3 is not None and not res3["continued"] and res3["n"] == 8 and res3["n_prior"] == 0,
        f"{res3['n']} evals from scratch")
    dsn.box["taper"] = list(BOUNDS[dsn.role]["taper"])
    dsn.budget = 16
    #  the wizard: the SOLVER step owns the run, CONVERGENCE the continuation
    dp._solver_rows = dp._conv_rows = None
    dp.nav.select("w.solver")
    solver_keys = [pm.key for pm in dp.rows().params]
    dp.nav.select("w.conv")
    conv_keys = [pm.key for pm in dp.rows().params]
    rep("the SOLVER step carries the optimiser, CONVERGENCE the keep-going rows",
        all(kk in solver_keys for kk in ("obj", "budget", "run"))
        and "go" in conv_keys and "more" in conv_keys and "run" not in conv_keys,
        f"solver {solver_keys[-3:]}, convergence {conv_keys[1:]}")
    rep("the wing search flew the END PLATE it was given, not a flat one",
        dsn.spec.plate_airfoil in g.lib.airfoils
        and dsn.spec.aero.get("plate_airfoil") == dsn.spec.plate_airfoil,
        f"plates {dsn.spec.plate_airfoil!r}")
    rep("the optimised wing is priced in lap time against the empty slot",
        dsn.lap is not None and dsn.lap.ok,
        f"lap {dsn.lap.time:.4f} s, {dsn.lap.time - dsn.base_lap.time:+.4f} s vs empty")
    key(pygame.K_s)
    rep("S saves the working wing under a new name (built-in origin)", g.build.left.wing == dsn.spec.name
        and dsn.spec.name in lib.wings and not lib.wings[dsn.spec.name].builtin and dsn.spec.name != "flank-e423",
        dsn.spec.name)
    # --- the airfoil page
    key(pygame.K_a)
    rep("A opens the airfoil page ranked", g.page == "airfoil" and len(g.af_page.list.items) >= 30,
        f"{len(g.af_page.list.items)} sections, top {g.af_page.list.items[0][0]}")
    t0 = time.perf_counter()
    g.frame(1.0 / 60.0)
    ms = (time.perf_counter() - t0) * 1e3
    rep("airfoil page draws", *_budget(ms, 80.0))
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_airfoil.png"))
    g.af_page.list.idx = 0
    top_name = g.af_page.list.items[0][0]
    key(pygame.K_RETURN)
    rep("ENTER assigns the section and returns to the DESIGN navigator",
        g.page == "section" and dsn.spec.airfoil == top_name, dsn.spec.airfoil)
    rep("and the AIRFOIL group re-seeded onto the section that was assigned",
        g.design_page.af.seed == top_name, g.design_page.af.seed)
    key(pygame.K_ESCAPE)
    #  ESC steps BACK through the chain rather than dropping to the car:
    #  design -> mission -> car. The wing is saved on the way out, which is
    #  the part that was load-bearing.
    rep("ESC leaves the design page, saves the dirty wing, and steps back to the mission",
        g.page == "mission" and lib.wings[dsn.spec.name].airfoil == top_name,
        f"{g.page}, section {lib.wings[dsn.spec.name].airfoil}")
    key(pygame.K_ESCAPE)
    rep("ESC again steps back to the car", g.page == "car", g.page)
    # --- the library page
    key(pygame.K_l)
    rep("L opens the library", g.page == "library" and len(g.lib_page.wings.items) >= 5, f"{len(g.lib_page.wings.items)} wings")
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
    rep("loading the build restores the car", g.build.left.wing == dsn.spec.name and g.build.top.wing == "rear-s1223", g.build.summary(lib)[:60])
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
    for _ in range(7):
        key(pygame.K_DOWN)
    a = key(pygame.K_RETURN)
    rep("menu 'Quit' returns quit", a == "quit" and not g.menu.open, str(a))
    shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
