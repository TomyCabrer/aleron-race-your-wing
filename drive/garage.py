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
  twist, end plates. The DESIGNER page edits one, runs the vortex lattice
  live (the AeroBO car-wing physics, numpy port) and can hand the planform
  to a Bayesian optimiser. The AIRFOIL page ranks the section library and
  runs XFOIL in the background. The LIBRARY page saves and loads wings
  and whole builds, so a wing drawn for one car goes on the next.
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
from .aero.wing import (WingSpec, BOUNDS, V_REF, ROLES, design_point, spanwise,
                        wing_cl, wing_cd, re_bank_snap)
from .aero import optimize as opt
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
PAGES = ("car", "design", "airfoil", "library")

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
    ("D / A / L", "designer / airfoils / library"),
    ("R / C", "car defaults / reset camera"),
    ("ENTER", "drive this car"),
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
    ("L3", "designer for this slot"),
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

    def hud_kwargs(self, lib: "Library") -> dict:
        w = self.wings(lib)
        wl, wr, wt = w["left"], w["right"], w["top"]
        ref = wl or wr
        out = dict(dev_left=wl is not None, dev_right=wr is not None,
                   x_w_left=round(self.left.x, 4), x_w_right=round(self.right.x, 4),
                   dev_chord=(ref.chord if ref is not None else DEV_CHORD),
                   dev_span=(ref.span if ref is not None else DEV_SPAN),
                   dev_plate=(ref.plate_h if ref is not None else 0.0),
                   wing_left_name=(wl.name if wl is not None else ""),
                   wing_right_name=(wr.name if wr is not None else ""),
                   top_on=wt is not None, top_x=round(self.top.x, 4),
                   top_span=(wt.span if wt is not None else 0.0),
                   top_chord=(wt.chord if wt is not None else 0.0),
                   top_plate=(wt.plate_h if wt is not None else 0.0),
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
        if int(d.get("version", 1)) < 2 and "slots" not in d:
            # a WingDesign file: the published panel on both flanks
            wd = WingDesign(wing=str(d.get("wing", "plate")), x_w=float(d.get("x_w", 0.97)),
                            h_w=float(d.get("h_w", 0.90)), inc_deg=float(d.get("inc_deg", 0.0))).clamp()
            name = wd.wing if wd.wing != "off" else ""
            b = cls(left=Slot(name, wd.x_w, wd.h_w, wd.inc_deg), right=Slot(name, wd.x_w, wd.h_w, wd.inc_deg))
            return b
        sl = d.get("slots", {})

        def _slot(k, default):
            v = sl.get(k, {})
            return Slot(str(v.get("wing", default.wing)), float(v.get("x", default.x)),
                        float(v.get("h", default.h)), float(v.get("inc_deg", default.inc_deg)),
                        str(v.get("mode", default.mode)))

        base = cls()
        return cls(name=str(d.get("name", "my corsa")), left=_slot("left", base.left),
                   right=_slot("right", base.right), top=_slot("top", base.top),
                   mirror=bool(d.get("mirror", True)), builtin=bool(d.get("builtin", False)))

    def save(self, path: str = DESIGN_PATH) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str = DESIGN_PATH) -> "CarBuild | None":
        try:
            with open(path) as f:
                return cls.from_json(json.load(f))
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def copy(self) -> "CarBuild":
        return CarBuild.from_json(self.to_json())


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


def build_car_mesh() -> list[tuple[np.ndarray, tuple, str]]:
    """(verts (n,3), colour, kind). Kind is 'body' | 'wheel' | 'trim'."""
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
                       or _between(x0, x1, X_REAR_GLASS) else C_PAINT)
            elif j in (2, 5):             # belt -> roof edge: side glass
                col = C_GLASS if _between(x0, x1, X_SIDE_GLASS) else C_PAINT
            elif j in (8, 9):             # floor
                col = C_UNDER
            else:
                col = C_PAINT
            polys.append((_orient(quad, inside), col, "body"))
    polys.append((_orient(rings[0], inside), C_PAINT_DARK, "body"))       # nose
    polys.append((_orient(rings[-1], inside), C_PAINT_DARK, "body"))      # tail

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
        self.car = Batch(build_car_mesh())
        self.n_polys = 0
        self.frame_ms = 0.0

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

    def _draw_info(self, build: CarBuild, lib: Library, key: str, deploy: float) -> None:
        u = self.ui
        r = self._panel((884, 12, 384, 470))
        x, y = r.x + 12 * u, r.y + 8 * u
        slot = build.slot(key)
        spec = lib.wings.get(slot.wing)
        on = spec is not None
        self._txt(SLOT_LABEL[key] + ("   (mirrored)" if build.mirror and key != "top" else ""),
                  x, y, self.f_lbl, C_PANEL_ON if on else C_TEXT_DIM)
        y += 20 * u
        self._txt((spec.name if on else "none")[:22], x, y, self.f_big, C_PANEL_ON if on else C_TEXT_DIM)
        y += 40 * u
        if on:
            src = "XFOIL" if not spec.aero.get("polar_is_estimate", True) else "estimate"
            geo = (f"{spec.airfoil}  b {spec.span:.2f} c {spec.chord:.2f} taper {spec.taper:.2f} "
                   f"S {spec.S:.2f} m2")
            if spec.legacy:
                geo = f"published panel S {spec.legacy.get('S', 0.35):.2f} m2 CL0 {spec.legacy['CL0']:.2f} L/D {spec.legacy['LD']:.1f}"
            rows = [(geo, C_TEXT_DIM)]
        else:
            rows = [("W: put a library wing in this slot   D: design one", C_TEXT_DIM)]
        rows += [
            (f"station x    {slot.x:+.2f} m", C_TEXT),
            (f"             {station_label(slot.x)}", C_TEXT_DIM),
            (f"height  h    {slot.h:.2f} m", C_TEXT),
            (f"incidence    {slot.inc_deg:+.0f} deg", C_TEXT),
        ]
        if key == "top":
            rows.append((f"deploys      {'brake + steer (active)' if slot.mode == 'active' else 'whenever armed (fixed)'}", C_TEXT))
        for s_, c in rows:
            self._txt(s_[:52], x, y, self.f_val if c is C_TEXT else self.f_lbl, c)
            y += 22 * u
        y += 6 * u
        if on and (spec.aero or spec.legacy):
            V = V_REF[spec.role]
            dp = design_point(spec, slot.inc_deg, V=V, x_w=slot.x)
            if dp:
                tag = "" if (spec.legacy or not spec.aero.get("polar_is_estimate", True)) else "  (ESTIMATE polar)"
                self._txt(f"AT {V:.1f} m/s{tag}", x, y, self.f_lbl, C_TEXT_DIM)
                y += 20 * u
                self._txt(f"CL {dp['CL']:.2f}  F {dp['F']:4.0f} N  D {dp['D']:3.0f} N  L/D {dp['LD']:.1f}",
                          x, y, self.f_val)
                y += 22 * u
                if spec.role == "flank":
                    self._txt(f"= {100 * dp['F'] / (CAR.m * G):.2f}% of mg  (x+b)/b x{(slot.x + CAR.b) / CAR.b:.2f}",
                              x, y, self.f_lbl, C_TEXT_DIM)
                    y += 22 * u
                    self._txt("corner-speed gain (crossover.gain)", x, y, self.f_lbl, C_TEXT_DIM)
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
                        self._txt(s_, x, y, self.f_val, c)
                        y += 22 * u
                else:
                    share_f = (slot.x + CAR.b) / CAR.L
                    self._txt(f"downforce split  front {100 * share_f:.0f}%  rear {100 * (1 - share_f):.0f}%",
                              x, y, self.f_lbl, C_OK if share_f > 0.3 else C_WARN)
                    y += 22 * u
                    self._txt(f"= {100 * dp['F'] / (CAR.m * G):.2f}% of mg; a front-limited car wants it forward",
                              x, y, self.f_lbl, C_TEXT_DIM)
                    y += 22 * u
                if dp.get("stalled"):
                    self._txt("STALLED at this incidence", x, y, self.f_val, C_WARN)
                    y += 22 * u
                else:
                    self._txt(f"stall margin {dp['stall_margin_deg']:.1f} deg", x, y, self.f_lbl,
                              C_OK if dp["stall_margin_deg"] > 2.0 else C_WARN)
                    y += 22 * u
        y = r.bottom - 26 * u
        self._txt(f"preview: {'DEPLOYED' if deploy > 0.5 else 'stowed'}   "
                  f"{build.summary(lib)[:38]}", x, y, self.f_lbl,
                  C_PANEL_ON if deploy > 0.5 else C_TEXT_DIM)

    def _draw_help(self, pad_name, hint, status="") -> None:
        u = self.ui
        r = self._panel((12, 690, 1256, 98))
        x, y = r.x + 10 * u, r.y + 6 * u
        lines = [
            ("GARAGE  -  three wings: flank left / right, top.  1 2 3 select, arrows place, W wing, "
             "D design, A airfoils, L library", C_TEXT),
            ("mouse drag orbit | wheel zoom | LEFT/RIGHT x | UP/DOWN h | [ ] incidence | M mirror | "
             "T top mode | SPACE deploy preview | R defaults | C camera", C_TEXT_DIM),
            ("ENTER drive  |  ESC menu (controls, defaults, quit)", C_TEXT_DIM),
        ]
        if pad_name:
            lines[2] = (f"PS5 {pad_name}:  L-stick move | R-stick orbit | L1/R1 incidence | "
                        "TRIANGLE slot | SQUARE wing | CIRCLE deploy | L3 design | CROSS drive | "
                        "OPTIONS menu", C_OK)
            lines.append(("ENTER drive  |  ESC menu (controls, defaults, quit)", C_TEXT_DIM))
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
OBJECTIVES = {
    "flank": ("corner gain @ drag cap", "force / drag @ force floor", "max force @ drag cap"),
    "top": ("downforce @ drag cap", "Fz / drag @ downforce floor"),
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
        self.drag_cap = 80.0 if self.role == "flank" else 60.0
        self.force_floor = 150.0 if self.role == "flank" else 300.0
        self.budget = 32
        self.result = None
        self.msg = ""
        self.polar = None
        self.dp = {}
        self.span_data = None
        self.err = ""
        self.params = ui.ParamList(self._build_params(), title=f"DESIGN  {SLOT_LABEL[key]}")
        self.update()

    # -- the parameter rows --------------------------------------------------
    def _set(self, attr, lo=None, hi=None):
        def f(v):
            setattr(self.spec, attr, float(v) if lo is not None else v)
            self.spec.clamp()
            self.dirty = True
            self.update()
        return f

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
            P("sec", "SECTION", None, kind="label"),
            P("airfoil", "airfoil", lambda: self.spec.airfoil, self._set("airfoil"), kind="choice",
              choices=names, help="LEFT/RIGHT cycles the section library; A opens it with the polar plots"),
            P("browse", "browse the airfoil library  (A)", None, lambda _: self.g.open_airfoils(), kind="action"),
            P("pf", "PLANFORM", None, kind="label"),
            P("span", "span" if self.role == "top" else "span (vertical)", lambda: self.spec.span, self._set("span", *b["span"]),
              step=0.02, fine=0.005, lo=b["span"][0], hi=b["span"][1], unit="m",
              help="the lattice's first-order variable: at fixed area span IS aspect ratio"),
            P("chord", "root chord", lambda: self.spec.chord, self._set("chord", *b["chord"]),
              step=0.02, fine=0.005, lo=b["chord"][0], hi=b["chord"][1], unit="m"),
            P("taper", "taper (tip/root)", lambda: self.spec.taper, self._set("taper", *b["taper"]),
              step=0.05, fine=0.01, lo=b["taper"][0], hi=b["taper"][1], fmt="{:.2f}"),
            P("twist", "tip twist", lambda: self.spec.twist_deg, self._set("twist_deg", *b["twist_deg"]),
              step=0.5, fine=0.1, lo=b["twist_deg"][0], hi=b["twist_deg"][1], unit="deg", fmt="{:+.1f}",
              help="washout (negative) unloads the tip: e up, stall margin up"),
            P("plate", "end plates", lambda: self.spec.plate_h, self._set("plate_h", *b["plate_h"]),
              step=0.01, fine=0.002, lo=b["plate_h"][0], hi=b["plate_h"][1], unit="m",
              help="tip plates in the lattice: cut induced drag, add wetted area"),
            P("mt", "MOUNT (this slot)", None, kind="label"),
            P("inc", "incidence", lambda: self.g.build.slot(self.key).inc_deg, self._slot_set("inc_deg"),
              step=0.5, fine=0.1, lo=b["inc_deg"][0], hi=b["inc_deg"][1], unit="deg", fmt="{:+.1f}",
              help="the angle the wing is bolted on at; the optimiser moves it too"),
            P("x", "station x", lambda: self.g.build.slot(self.key).x, self._slot_set("x"),
              step=0.05, fine=0.01, lo=CAR_X_REAR, hi=CAR_X_FRONT, unit="m", fmt="{:+.2f}",
              help="forward of the CG: (x + b)/b multiplies the flank gain; a top wing behind the rear axle unloads the front"),
            P("h", "height h", lambda: self.g.build.slot(self.key).h, self._slot_set("h"),
              step=0.05, fine=0.01, lo=H_W_MIN if self.role == "flank" else 0.9,
              hi=H_W_MAX if self.role == "flank" else TOP_H_MAX, unit="m",
              help="a top wing's ride height sets its ground effect; the flank's h sets the roll arm"),
        ]
        if self.role == "top":
            rows.append(P("mode", "deploys", lambda: self.g.build.slot(self.key).mode, self._slot_set("mode"),
                          kind="choice", choices=["active", "fixed"],
                          help="active = out under brake or steering, stowed on the straights"))
        rows += [
            P("op", "OPTIMISER (GP Bayesian, AeroBO)", None, kind="label"),
            P("obj", "objective", lambda: self.objective, self._set_attr("objective"), kind="choice",
              choices=list(OBJECTIVES[self.role])),
            P("cap", "drag cap", lambda: self.drag_cap, self._set_attr("drag_cap"), step=5.0, fine=1.0,
              lo=5.0, hi=400.0, unit="N", fmt="{:.0f}", help="at the design speed"),
            P("floor", "force floor", lambda: self.force_floor, self._set_attr("force_floor"), step=10.0, fine=2.0,
              lo=10.0, hi=2000.0, unit="N", fmt="{:.0f}"),
            P("budget", "evaluations", lambda: self.budget, self._set_attr("budget"), kind="int", lo=12, hi=96),
            P("run", "run the optimiser  (O)", None, lambda _: self.optimise(), kind="action"),
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

    # -- analysis --------------------------------------------------------------
    @property
    def slot(self) -> Slot:
        return self.g.build.slot(self.key)

    def update(self) -> None:
        slot = self.slot
        ride = slot.h if self.role == "top" else None
        aero = self.lib.analyse_wing(self.spec, ride_h=ride, standoff=DEV_OUT0 + DEV_OUT1)
        self.err = aero.get("error", "")
        self.polar = self.lib.wing_polar(self.spec)
        self.dp = design_point(self.spec, slot.inc_deg, x_w=slot.x) if "CLa" in aero else {}
        try:
            self.span_data = spanwise(self.spec, self.polar, slot.inc_deg, ride_h=ride)
        except ValueError:
            self.span_data = None
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
        if spec.name in self.lib.wings and self.lib.wings[spec.name].builtin:
            spec.name = self.lib.unique_name("wings", spec.name)
        spec.builtin = False
        spec.legacy = None
        self.lib.save_wing(spec.copy())
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
    def _objective_value(self, dp: dict) -> float:
        F, D = dp["F"], dp["D"]
        obj = self.objective
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

    def optimise(self) -> dict | None:
        slot = self.slot
        role = self.role
        b = BOUNDS[role]
        ride = slot.h if role == "top" else None
        polar = self.polar
        if polar is None:
            return None
        from .aero.wing import analyse as _analyse
        span_hi = b["span"][1]
        if role == "flank":                       # fit between sill and roof
            span_hi = min(span_hi, 2.0 * min(slot.h - 0.28, 1.34 - slot.h))
        bounds = [(b["span"][0], max(span_hi, b["span"][0] + 0.05)), b["chord"], b["taper"],
                  b["twist_deg"], b["plate_h"], b["inc_deg"]]
        x0 = [self.spec.span, self.spec.chord, self.spec.taper, self.spec.twist_deg,
              self.spec.plate_h, slot.inc_deg]
        V = V_REF[role]

        def f(x):
            sp = self.spec.copy(span=float(x[0]), chord=float(x[1]), taper=float(x[2]),
                                twist_deg=float(x[3]), plate_h=float(x[4]))
            if role == "flank" and (slot.x + 0.5 * sp.chord > CAR_X_FRONT or slot.x - 0.5 * sp.chord < CAR_X_REAR):
                return -math.inf
            try:
                sp.aero = _analyse(sp, polar, V=V, ride_h=ride, standoff=DEV_OUT0 + DEV_OUT1)
            except ValueError:
                return -math.inf
            dp = design_point(sp, float(x[5]), V=V, x_w=slot.x)
            if not dp or dp.get("stalled") or dp["stall_margin_deg"] < 2.0:
                return -math.inf
            return self._objective_value(dp)

        n = int(self.budget)
        t0 = time.perf_counter()
        bo = opt.maximise(f, bounds, n_init=min(8, n // 2), n_iter=max(n - min(8, n // 2), 1), seed=0, x0=x0)
        rs = opt.random_search(f, bounds, n=n, seed=1)
        dt = time.perf_counter() - t0
        f0 = f(np.asarray(x0))
        self.result = dict(bo=bo["f_best"], rs=rs["f_best"], start=f0, trace=bo["best_trace"],
                           rs_trace=rs["best_trace"], n=n, secs=dt, objective=self.objective,
                           x=bo["x_best"].tolist())
        if math.isfinite(bo["f_best"]) and bo["f_best"] >= f0:
            x = bo["x_best"]
            self.spec.span, self.spec.chord, self.spec.taper = float(x[0]), float(x[1]), float(x[2])
            self.spec.twist_deg, self.spec.plate_h = float(x[3]), float(x[4])
            self.spec.clamp()
            slot.inc_deg = float(x[5])
            self.g.build.sync_mirror(self.key)
            self.dirty = True
            self.msg = (f"BO {bo['f_best']:.3g} vs random {rs['f_best']:.3g} vs start {f0:.3g} "
                        f"({n} evals, {dt:.1f} s) - applied")
        else:
            self.msg = f"no feasible design found under the constraints ({n} evals); try a wider cap"
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
        if self.result:
            rr = self.result
            text.blit(screen, f"optimiser: {rr['objective']}  BO {rr['bo']:.4g}  random {rr['rs']:.4g}  "
                      f"start {rr['start']:.4g}   {rr['n']} evals {rr['secs']:.1f} s", x, y, 13, ui.C_SECTION)
            y += 18
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
            P("cl_max", "cl_max", lambda: self.weights["cl_max"], self._w("cl_max"), step=0.05, lo=0, hi=1),
            P("ld_max", "L/D max", lambda: self.weights["ld_max"], self._w("ld_max"), step=0.05, lo=0, hi=1),
            P("thin", "thickness (thin)", lambda: self.weights["thin"], self._w("thin"), step=0.05, lo=0, hi=1),
            P("cm", "low |cm|", lambda: self.weights["cm"], self._w("cm"), step=0.05, lo=0, hi=1),
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
        ms = [self.metrics(n) for n in sorted(self.lib.airfoils)]
        ok = [m for m in ms if m.get("polar") is not None]

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
        for m in ms:
            m.setdefault("score", 0.0)
        keyf = {"score": lambda m: -m["score"], "cl_max": lambda m: -m["cl_max"], "ld_cr": lambda m: -m["ld_cr"],
                "name": lambda m: m["name"], "tc": lambda m: m["tc"]}[self.sort]
        ms.sort(key=keyf)
        items = []
        for m in ms:
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
        self.af_page = AirfoilPage(self)
        self.lib_page = LibraryPage(self)
        self.prompt = ui.TextPrompt()
        self._prompt_kind = ""
        self._af_return = "design"
        self.status = ""

    # the old attribute name
    @property
    def design(self) -> CarBuild:
        return self.build

    # -- pause menu (ESC / OPTIONS) -------------------------------------------
    def _menu_open(self) -> None:
        secs = [("KEYBOARD", GARAGE_HELP_KB)]
        note = ""
        if self.pad is not None:
            secs.append(("PS5 DUALSENSE" if self.pad.layout == "ps" else "GAMEPAD",
                         GARAGE_HELP_PAD))
        else:
            note = "no controller: pair the DualSense (CREATE+PS) - it hot-plugs"
        self.menu.show(
            items=[("Resume", "resume"),
                   (f"Design the {self.sel} wing", "design"),
                   ("Airfoil library", "airfoils"),
                   ("Wing & build library", "library"),
                   ("Reset car to defaults", "defaults"),
                   ("Reset camera", "camera"),
                   ("Drive this car", "drive"),
                   ("Quit", "quit")],
            sections=secs, note=note,
            subtitle=self.build.summary(self.lib)[:120],
            footer="ESC / OPTIONS resume   R defaults   C camera   ENTER / CROSS select")
        self._menu_stick = StickNav()

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
            self.open_designer()
        elif action == "airfoils":
            self.open_airfoils()
        elif action == "library":
            self.open_library()
        elif action in ("drive", "quit"):
            return action
        return None

    # -- pages ------------------------------------------------------------------
    def open_designer(self, key: str | None = None) -> None:
        if key is not None:
            self.sel = key
        if self.designer is None or self.designer.key != self.sel:
            self.designer = Designer(self, self.sel)
        self.page = "design"

    def open_airfoils(self) -> None:
        if self.designer is None:
            self.open_designer()
        d = self.designer
        self._af_return = "design"
        cl = d.dp.get("CL", 0.8) if d.dp else 0.8
        self.af_page.open(d.spec.reynolds(), cl_design=abs(cl) + 0.2, keep=d.spec.airfoil)
        self.page = "airfoil"

    def open_library(self) -> None:
        self.lib_page.refresh()
        self.page = "library"

    def close_page(self) -> None:
        if self.page == "design" and self.designer is not None and self.designer.dirty:
            self.designer.commit()
            self.hint = self.designer.msg
        if self.page == "airfoil":
            self.page = self._af_return
            return
        self.page = "car"

    def assign_airfoil(self) -> None:
        name = self.af_page.current_name()
        if name and self.designer is not None:
            self.designer.spec.airfoil = name
            self.designer.dirty = True
            self.designer.update()
            self.hint = f"section {name} on {self.designer.spec.name}"
            self.page = "design"

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
                self.lib.save_airfoil(AirfoilSpec(name, "naca", code=code, notes="user NACA 4-digit"))
            self.af_page.open(self.af_page._re, keep=name)
            self.hint = f"section {name} added"
        elif kind == "build":
            self.build.name = value[:32]
            b = self.build.to_json()
            self.lib.save_build(b)
            self.lib_page.refresh()
            self.hint = f"build '{value}' saved to the library"

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
                self.open_designer()
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
            if self.page == "design" and self.designer is not None:
                self._run_optimiser()
            elif self.page == "airfoil":
                self.af_page.request_xfoil()
            elif self.page == "library":
                self._save_build_quick()
        if e["triangle"]:
            if self.page == "design":
                self.open_airfoils()
            elif self.page == "library":
                self.lib_page.focus = "builds" if self.lib_page.focus == "wings" else "wings"
            elif self.page == "airfoil":
                self.af_page.focus = "params" if self.af_page.focus == "list" else "list"
        if e["options"]:
            self._menu_open()
        return None

    # -- page-generic navigation ------------------------------------------------
    def _page_nav(self, d: int) -> None:
        if self.page == "design" and self.designer is not None:
            self.designer.params.nav(d)
        elif self.page == "airfoil":
            if self.af_page.focus == "list":
                self.af_page.list.nav(d)
            else:
                self.af_page.params.nav(d)
        elif self.page == "library":
            (self.lib_page.wings if self.lib_page.focus == "wings" else self.lib_page.builds).nav(d)

    def _page_adjust(self, d: int, fine: bool) -> None:
        if self.page == "design" and self.designer is not None:
            self.designer.params.adjust(d, fine)
        elif self.page == "airfoil" and self.af_page.focus == "params":
            self.af_page.params.adjust(d, fine)
        elif self.page == "airfoil":
            self.af_page.list.nav(d)

    def _page_activate(self) -> None:
        if self.page == "design" and self.designer is not None:
            self.designer.params.activate()
        elif self.page == "airfoil":
            if self.af_page.focus == "list":
                self.assign_airfoil()
            else:
                self.af_page.params.activate()
        elif self.page == "library":
            self._library_select()

    def _run_optimiser(self) -> None:
        if self.designer is None:
            return
        self.hint = "optimising..."
        self._draw_page()
        pygame.display.flip()
        self.designer.optimise()
        self.hint = "optimiser done"

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
        name = self.build.name if self.build.name not in self.lib.builds else self.lib.unique_name("builds", self.build.name)
        self.build.name = name
        self.lib.save_build(self.build.to_json())
        self.lib_page.refresh()
        self.hint = f"build '{name}' saved"

    def _new_wing(self) -> None:
        slot = self.build.slot(self.sel)
        slot.wing = ""
        self.build.sync_mirror(self.sel)
        self.designer = None
        self.open_designer()

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
            return None                                 # mouse etc. ignored
        if ev.type == pygame.KEYDOWN:
            if self.page == "car":
                return self._handle_car_key(ev)
            return self._handle_page_key(ev)
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
        elif k == pygame.K_d:
            self.open_designer()
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
        elif k == pygame.K_LEFT:
            self._page_adjust(-1, fine)
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._page_activate()
        elif k == pygame.K_TAB:
            if self.page == "library":
                self.lib_page.focus = "builds" if self.lib_page.focus == "wings" else "wings"
            elif self.page == "airfoil":
                self.af_page.focus = "params" if self.af_page.focus == "list" else "list"
        elif self.page == "design" and self.designer is not None:
            if k == pygame.K_o:
                self._run_optimiser()
            elif k == pygame.K_s:
                self.designer.commit()
                self.hint = self.designer.msg
            elif k == pygame.K_a:
                self.open_airfoils()
            elif k == pygame.K_x:
                self.designer.request_xfoil()
                self.hint = self.designer.msg
            elif k == pygame.K_n:
                self.prompt_rename()
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
        if self.page == "design" and self.designer is not None:
            help_ = self.designer.draw(self.screen, self.text, self.plot, u)
            hints = [("UP/DOWN", "row"), ("LEFT/RIGHT", "adjust (SHIFT fine)"), ("ENTER", "action"),
                     ("O", "optimise"), ("S", "save + use"), ("A", "airfoils"), ("X", "XFOIL"), ("N", "rename"),
                     ("ESC", "back (saves)")]
            pad_h = [("stick/d-pad", "row / adjust"), ("L1", "fine"), ("CROSS", "action"), ("SQUARE", "optimise"),
                     ("TRIANGLE", "airfoils"), ("CIRCLE", "back")] if pad_name else None
        elif self.page == "airfoil":
            help_ = self.af_page.draw(self.screen, self.text, self.plot, u)
            hints = [("UP/DOWN", "section"), ("TAB", "list / weights"), ("ENTER", "use this section"),
                     ("X", "XFOIL polar"), ("N", "new NACA"), ("ESC", "back")]
            pad_h = [("stick", "section"), ("CROSS", "use"), ("SQUARE", "XFOIL"), ("TRIANGLE", "list/weights"),
                     ("CIRCLE", "back")] if pad_name else None
        else:
            self.lib_page.draw(self.screen, self.text, u)
            hints = [("UP/DOWN", "item"), ("TAB", "wings / builds"), ("ENTER", "use / load"),
                     ("S", "save car as build"), ("N", "new wing"), ("DEL", "delete"), ("ESC", "back")]
            pad_h = [("stick", "item"), ("TRIANGLE", "wings/builds"), ("CROSS", "use / load"),
                     ("SQUARE", "save build"), ("CIRCLE", "back")] if pad_name else None
        title = {"design": "DESIGNER", "airfoil": "AIRFOIL LIBRARY", "library": "WING & BUILD LIBRARY"}[self.page]
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
                        self.lib.save_wing(w)
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

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    g = Garage((1280, 800), b.copy(), headless=True, lib=lib)
    t0 = time.perf_counter()
    for i in range(12):
        g.deploy_cmd = 1.0 if i >= 4 else 0.0
        g.frame(1.0 / 60.0)
    ms = (time.perf_counter() - t0) * 1e3 / 12.0
    rep("car page frame budget", ms < 14.0, f"{ms:.2f} ms/frame, {g.view.n_polys} polys drawn (3 wings)")
    rep("deploy preview animates", 0.0 < g.deploy <= 1.0, f"deploy {g.deploy:.2f}")
    if screenshot_dir:
        os.makedirs(screenshot_dir, exist_ok=True)
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_frame.png"))

    def key(k, mod=0):
        return g._handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod))

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
    rep("D opens the designer on the selected slot", g.page == "design" and g.designer is not None
        and g.designer.key == "left" and g.designer.spec.name == "flank-e423", f"{g.page} {g.designer.spec.name if g.designer else None}")
    t0 = time.perf_counter()
    g.frame(1.0 / 60.0)
    g.frame(1.0 / 60.0)
    ms = (time.perf_counter() - t0) * 1e3 / 2
    rep("designer page draws", ms < 60.0, f"{ms:.1f} ms/frame")
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_design.png"))
    dsn = g.designer
    dsn.params.select_key("span")
    s0 = dsn.spec.span
    key(pygame.K_RIGHT)
    rep("RIGHT on the span row widens the wing and re-analyses", abs(dsn.spec.span - s0 - 0.02) < 1e-9
        and dsn.dirty and "CLa" in dsn.spec.aero, f"span {s0:.3f} -> {dsn.spec.span:.3f}, CLa {dsn.spec.aero.get('CLa', 0):.3f}")
    dsn.params.select_key("inc")
    i0 = g.build.left.inc_deg
    key(pygame.K_RIGHT)
    rep("incidence row edits the slot (mirrored)", abs(g.build.left.inc_deg - i0 - 0.5) < 1e-9
        and abs(g.build.right.inc_deg - g.build.left.inc_deg) < 1e-9, f"inc {g.build.left.inc_deg:+.1f}")
    dsn.budget = 16
    t0 = time.perf_counter()
    res = dsn.optimise()
    dt = time.perf_counter() - t0
    rep("optimiser runs (16 evals) and reports BO vs random", res is not None and res["n"] == 16 and dt < 20.0
        and math.isfinite(res["bo"]), f"BO {res['bo']:.3f} random {res['rs']:.3f} start {res['start']:.3f} in {dt:.1f} s")
    rep("optimised design respects the drag cap", dsn.dp and dsn.dp["D"] <= dsn.drag_cap + 1e-6,
        f"D {dsn.dp['D']:.1f} N <= {dsn.drag_cap:.0f} N")
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
    rep("airfoil page draws", ms < 80.0, f"{ms:.1f} ms")
    if screenshot_dir:
        pygame.image.save(g.screen, os.path.join(screenshot_dir, "garage_airfoil.png"))
    g.af_page.list.idx = 0
    top_name = g.af_page.list.items[0][0]
    key(pygame.K_RETURN)
    rep("ENTER assigns the section and returns to the designer", g.page == "design" and dsn.spec.airfoil == top_name, dsn.spec.airfoil)
    key(pygame.K_ESCAPE)
    rep("ESC leaves the designer and saves the dirty wing", g.page == "car" and lib.wings[dsn.spec.name].airfoil == top_name, "")
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
    rep("L3 opens the designer on the pad", g2.page == "design", g2.page)
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
