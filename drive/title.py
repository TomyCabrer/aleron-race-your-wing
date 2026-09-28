"""drive/title.py -- the title screen: the owner's ALERON logo over a live 3-D scene (task 44).

The owner (2026-09-25): "We need a proper title screen, with a background",
and of the background: "Randomised the type and number of cars (1 per type
of car)".

WHEN. Every interactive launch -- a player session (`drive._player_session`)
with no mode flag (`drive.TITLE_SKIP`: --garage, --race, --seed-lap,
--ml-drive, a swarm, a script, ...) -- opens here, after the settings and the
progress are loaded and before the first session (`drive.run_interactive_cli`).
The screen owns the window (`set_mode`, caption `CAPTION`) and reads the menu
commands the pause menu reads, from the same input stack (`input.BlendedInput`
in menu mode: the arrows, ENTER, ESC, the mouse, a pad), and hands back ONE
action, which the loop runs with what it already has:

    drive        today's first session (WELCOME / TIME TRIAL as always)
    challenges   a session that opens on the pause menu's Challenges page
    garage       the garage (BACKSPACE / the touchpad also pick it)
    tutorial     a session that opens on the Tutorial page
    settings     a session that opens on the Settings page
    quit         exit 0 (Quit picked, or the window's close; ESC, a right
                 click, CIRCLE or OPTIONS only move the cursor to Quit, task 45)

The title comes back whenever a session ends with 'title' (task 45): the
pause page's Main menu row, the garage's, or Back / ESC on a page the title
opened (drive.run_interactive_cli runs its pick as it does at launch).

THE BACKGROUND. A live scene drawn by the game's own renderer: a random dressed
circuit (`track.CIRCUITS`, never the skidpad or the dragstrip) and 1 to 5 cars
of DISTINCT types -- at most one per `cars.CAR_ORDER` entry, the owner's rule
-- each replaying its REFERENCE LAP (`medals.reference_trace`: the author lap
the medals are measured from, key `<map>|<car>|<engine>|none`, the first of
ENGINE_PREF that exists) in its own body and paint. A chase camera follows one
car and cuts to the next every CAM_SWITCH_S s. No physics runs: the laps are
replays, round and round (a flying lap ends where it starts, at speed).

THE WINGS. The owner (2026-09-26): "In the title screen they should be
deployed with all the wings." Every scene car carries the challenges' FULL
WING set (`WINGS_CONFIG`: the stock top wing and the stock side plates,
fitted to that car by `challenges.config_build`, as a challenge with no
build of the player's lends them; `wing_hud`). The owner (2026-09-27): "they
are always with the wings deployed, should only deploy in the curve." So the
wings WORK as the game's AUTO mode works them (`auto_deploy`, the
vehicle's own law on the replay): in a corner the OUTER flank's panel and the
top wing come out, on the straights they stow, with the car's ramp and hold
times. A replay has no steering wheel, so "in a corner" is its lateral
acceleration past AY_ON (AY_OFF to let go). Drawn only: no physics runs, so
the laps are the dry reference laps as before.

Why the cars run as a TRAIN (GAP_M apart, re-spaced at every cut) and not an
even fifth of a lap apart: with an even spread the nearest other car is 220-380
m away -- a speck, or out of sight -- and the chase view shows one car on an
empty road. In a train the camera's car sees the others ahead of it, down the
straight and into the corners. They are re-spaced at every camera cut (the cut
hides it), the gap in front of a car that is faster than the car ahead widened
by the lap-mean closing speed over a camera shot (GAP_MARGIN), so no car ever
drives through another in view; the train starts at a random point of the lap.

Why `_SceneRenderer` (a `render.Renderer` subclass): the renderer draws ONE
full car -- the driven car, in the module's fitted car and paint
(`render.set_car` / `set_paint`) -- and any other as a ghost in that same body.
The title wants each car in its OWN body and paint, so the car pass of the
frame (after the far props, before the near ones) draws every scene car in
view, far to near, pointing the module's car and paint at each in turn and
restoring them after. Each is the driven car's own drawing: mesh, shading,
shadow, a body roll from its lateral g, the front wheels steered by its yaw
rate, spinning rims and brake lamps from its deceleration. No HUD.

FALLBACK: no circuit with a reference lap, a renderer that cannot be built or
throws, or a machine too slow for the scene (median frame over SLOW_MS for
SLOW_FRAMES frames, a real window only): a slowly panned panorama
(`world.build_panorama`) under the same overlay. Every Graphics setting draws
the live scene (Classic without scenery, Low with fewer props).
"""
from __future__ import annotations

import gc
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import pygame

from . import render as rnd
from .menu import CLICK_GUARD_DRAWS, FONT_NAMES

# ==================================================================== #
#  CONSTANTS                                                           #
# ==================================================================== #
#: the palette (menu.py / garage_ui.py / render.py agree on these)
C_BG = (27, 29, 33)
C_TEXT = (232, 234, 238)
C_DIM = (139, 144, 153)
C_ACCENT = (255, 140, 43)
C_SEL_BG = (40, 44, 52)
C_KEY = (217, 206, 85)
C_SHADE = (9, 10, 13)            # the gradients laid over the scene

#: the menu column, top to bottom: (label, action). The action is what
#: `Title.run` returns; drive.run_interactive_cli does the rest
ITEMS = (("Drive", "drive"), ("Challenges", "challenges"), ("Garage", "garage"),
         ("Tutorial", "tutorial"), ("Settings", "settings"), ("Quit", "quit"))
ACTIONS = tuple(a for _, a in ITEMS)
#: the actions that are a session opened on a pause-menu page (Sim.open_page)
PAGES = ("challenges", "tutorial", "settings")
#: one line under the menu: what the highlighted row does
HINTS = {
    "drive": "your car on your map: laps, the time trial, the wings",
    "challenges": "three stars each: you pick the car and the wings",
    "garage": "build the car: the side wings and the top wing, in 3-D",
    "tutorial": "learn it step by step: the car, then the wings",
    "settings": "map, car, paint, engine, gearbox, aids, camera, sound",
    "quit": "back to the desktop",
}
#: the game's name (the owner, 2026-09-27): "Alerón: Race Your Wing". The logo
#: and the line under it are the owner's own art (2026-09-28: their
#: myTitleV1.pdf and RaceYourWing.pdf, cut out of the white page): ALERON in
#: blue pixel letters, an orange wing out of each side, over a blue rule, and
#: RACE YOUR WING in blue and orange, fitted under that rule
LOGO = "ALERÓN"
SUBTITLE = "RACE YOUR WING"
CAPTION = "Alerón: Race Your Wing"
ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "art")
LOGO_ART = os.path.join(ART_DIR, "logo.png")        # 833 x 150
SUB_ART = os.path.join(ART_DIR, "subtitle.png")     # 1273 x 116
LOGO_W = 560                     # the logo's width, px at 1280x800 (the art's own 833: sharp to u 1.5)
LOGO_RULE = (0.094, 0.923)       # the art's rule under the letters, fractions of its width
SUB_SIZE = 22                    # the plain subtitle's px at 1280x800 (no art)
#: the art's blue (the plain logo, the store art's rule and glow)
C_LOGO_BOT = (40, 92, 226)
FOOTER = "UP / DOWN move   ENTER / CROSS select   ESC to Quit   or click a row"

#: the scene. A reference lap's engine: the first of these that has one
ENGINE_PREF = ("sport", "tuned", "stock")
SURFACE = "none"                 # dry: the traces are the dry classes' author laps
CAM_SWITCH_S = 10.0              # s   the chase camera cuts to the next car
FADE_S = 0.5                     # s   the fade in after a cut and at the start
#: m, the gap to the car ahead in the train (uniform in this range), plus the
#: lap-mean closing speed x a camera shot x GAP_MARGIN when the car behind is
#: the faster: a lap's local speeds swing about the mean
GAP_M = (28.0, 70.0)
GAP_MARGIN = 1.5
FAR_M = 350.0                    # m   a car deeper than this is under 4 px: not drawn
ROLL_PER_G = 0.07                # rad of body roll per g of lateral acceleration: the
#                                  physics' ~4 deg at 1 g (render._car_frame3); a look only
BRAKE_AX = -2.5                  # m/s^2: a replay slowing harder than this lights its lamps
WINGS_CONFIG = "full"            # every scene car's wings: challenges' FULL WING
#: the wings in the corners only (`auto_deploy`): a replay is "in a corner" at
#: a lateral acceleration past AY_ON and leaves it under AY_OFF (m/s^2; the
#: straights sit under 1, the corners at 4-9); the ramps and holds are the
#: vehicle's (drive/vehicle.py: DEV_HOLD, DEV_DEP_LOCKOUT, TOP_HOLD, t_ext / t_ret)
AY_ON = 3.0
AY_OFF = 2.0
#: the drawn travel in steps of 1/DEP_STEPS: each step is a new wing soup
#: (~2.5 ms), so a 0.45 s ramp costs 10 rebuilds a car, not one a frame
DEP_STEPS = 10
SMOOTH_N = 5                     # samples (0.25 s at 20 Hz) of the yaw rate / accel smoothing
#: the panorama fallback: its pan (px/s at the window's lens) and its horizon
PAN_PX_S = 14.0
HORIZON_FRAC = 0.56
#: a machine too slow for the live scene (a real window only): the median
#: frame over this many frames above this -> the panorama
SLOW_MS = 40.0
SLOW_FRAMES = 60


# ==================================================================== #
#  THE SCENE: which map, which cars                                    #
# ==================================================================== #
def scene_maps() -> tuple:
    """The dressed circuits a scene may be on (never the skidpad, the
    dragstrip or the open ground)."""
    from . import track as trk
    return tuple(trk.CIRCUITS)


def _ref_trace(key: str):
    from .medals import reference_trace
    return reference_trace(key)


def available(track: str, trace=None) -> dict:
    """{car: (engine, trace)}: every car in `cars.CAR_ORDER` with a reference
    lap on `track` (dry), with the first engine of ENGINE_PREF that has one."""
    import cars
    from .records import class_key
    trace = trace or _ref_trace
    out = {}
    for car in cars.CAR_ORDER:
        for eng in ENGINE_PREF:
            a = trace(class_key(track, car, eng, SURFACE))
            if a is not None and np.ndim(a) == 2 and len(a) >= 8 and np.shape(a)[1] >= 9:
                out[car] = (eng, a)
                break
    return out


def pick_scene(seed=None, trace=None, maps=None):
    """A random scene: dict(track, cars=[(car, engine, trace), ...]) or None
    when no circuit has a reference lap for any car. The map is random among
    those with a lap; the NUMBER of cars random in 1..(cars with a lap there),
    at most one car per type (the owner's "1 per type of car")."""
    import cars
    rng = np.random.default_rng(seed)
    maps = list(maps if maps is not None else scene_maps())
    for i in rng.permutation(len(maps)):
        tr = maps[int(i)]
        av = available(tr, trace)
        if not av:
            continue
        keys = [k for k in cars.CAR_ORDER if k in av]
        n = int(rng.integers(1, len(keys) + 1))
        chosen = [keys[int(j)] for j in rng.choice(len(keys), size=n, replace=False)]
        return dict(track=tr, cars=[(k, av[k][0], av[k][1]) for k in chosen])
    return None


def wing_hud(key: str, lib) -> dict:
    """The HudData wing fields that draw `key`'s car with the FULL WING set
    (the module docstring's THE WINGS), every panel out; {} -- the car drawn
    bare -- when that set cannot be built. `ReplayCar.at` moves the panels
    in and out with the corners (`auto_deploy`)."""
    try:
        from .challenges import config_build
        b = config_build(None, lib, key, WINGS_CONFIG)
        kw = b.cfg_kwargs(lib)
        out = dict(b.hud_kwargs(lib))
        out.update(h_w=float(kw.get("h_w", rnd.H_W)),
                   inc_deg=math.degrees(float(kw.get("delta_dev_geom", 0.0))),
                   wing_on=True, wing_deploy=1.0,
                   wing_deploy_l=1.0 if out["dev_left"] else 0.0,
                   wing_deploy_r=1.0 if out["dev_right"] else 0.0,
                   top_deploy=1.0 if out["top_on"] else 0.0)
        return out
    except Exception as exc:              # noqa: BLE001 -- a title never stops a launch
        print(f"title: the {key} without wings ({type(exc).__name__}: {exc})")
        return {}


def _smooth(v: np.ndarray, n: int = SMOOTH_N) -> np.ndarray:
    if n <= 1 or len(v) < n:
        return v
    return np.convolve(v, np.ones(n) / n, mode="same")


def _smooth_lap(v: np.ndarray, n: int = SMOOTH_N) -> np.ndarray:
    """`_smooth` round a CLOSED lap: its last sample and its first are the
    same point of the road, so the window runs on across the line (the
    zero padding of `_smooth` pulls the ends toward 0 -- the wings let go
    of a corner that runs through the line; review 3)."""
    v = np.asarray(v, dtype=np.float64)
    k = n // 2
    if n <= 1 or len(v) < 2 * k + 2:
        return _smooth(v, n)
    ext = np.concatenate([v[-(k + 1):-1], v, v[1:k + 1]])
    return np.convolve(ext, np.ones(n) / n, mode="valid")[:len(v)]


def auto_deploy(t, ay) -> tuple:
    """The wings of a replay through its lap, as the game's AUTO mode moves
    them (`vehicle.Vehicle._aero`): (dep, side, top) arrays, one per sample
    -- the flank panel's travel 0..1 (smoothstepped, as drawn), the latched
    side (+1 a LEFT-hander: the RIGHT, outer, panel is out; HudData's
    `wing_side`) and the top wing's travel.

    The same law with the lateral acceleration standing in for the steering
    (a replay has none): a corner is |ay| past AY_ON, held until under AY_OFF;
    the side latches after DEV_HOLD s and never while the panel is out past
    DEV_DEP_LOCKOUT; the panel is armed while the corner turns the latched
    way; the top wing is out in a corner and TOP_HOLD s after it. The lap is
    run twice and the second pass kept: a flying lap ends where it starts, so
    the state at its line is the state the lap before left there."""
    from . import vehicle as veh
    t = np.asarray(t, dtype=np.float64)
    ay = np.asarray(ay, dtype=np.float64)
    n = len(t)
    dep, side, top = np.zeros(n), np.zeros(n), np.zeros(n)
    if n < 2:
        return dep, side, top
    dts = np.diff(t, prepend=t[0])
    dts[0] = float(np.median(dts[1:]))     # the wrap: one sample on
    t_ext, t_ret = veh.VehicleConfig.t_ext, veh.VehicleConfig.t_ret      # the flank panel
    tt_ext, tt_ret = veh.TopAero.t_ext, veh.TopAero.t_ret                # the top wing
    s_side, hold, raw, raw_t, hold_t, curving = 0, 0.0, 0.0, 0.0, 0.0, False
    for rec in (False, True):
        for i in range(n):
            dt, a = float(dts[i]), float(ay[i])
            curving = abs(a) > (AY_OFF if curving else AY_ON)
            want = (1 if a > 0.0 else -1) if curving else 0
            if want != 0 and want != s_side:
                hold += dt
                if hold >= veh.DEV_HOLD and raw <= veh.DEV_DEP_LOCKOUT:
                    s_side, hold = want, 0.0
            else:
                hold = 0.0
            cmd = 1.0 if (want != 0 and want == s_side) else 0.0
            raw = min(cmd, raw + dt / t_ext) if cmd > raw else max(cmd, raw - dt / t_ret)
            if want != 0:
                hold_t = veh.TOP_HOLD
            elif hold_t > 0.0:
                hold_t = max(0.0, hold_t - dt)
            cmd_t = 1.0 if (want != 0 or hold_t > 0.0) else 0.0
            raw_t = (min(cmd_t, raw_t + dt / tt_ext) if cmd_t > raw_t
                     else max(cmd_t, raw_t - dt / tt_ret))
            if rec:
                dep[i], side[i], top[i] = (veh._smoothstep(raw), s_side,
                                           veh._smoothstep(raw_t))
    return dep, side, top


class ReplayCar:
    """One scene car: its reference lap as arrays, replayed round and round
    from `t0`, and what the renderer needs to draw it at any time (a state
    with its pose, speed, yaw rate, roll and accelerations; its steered front
    wheels; its brake pedal; its wings: `wings`, wing_hud's fields, moved in
    and out with the corners by `auto_deploy`)."""

    def __init__(self, key: str, engine: str, trace, paint=None, wings=None):
        import cars
        a = np.asarray(trace, dtype=np.float64)
        self.key, self.engine = key, engine
        t = a[:, 0] - a[0, 0]
        #  a repeated stamp would make the gradients below divide by zero
        keep = np.concatenate([[True], np.diff(t) > 1e-9])
        a, t = a[keep], t[keep]
        self.t = t
        self.T = float(t[-1])
        self.x, self.y, self.psi = a[:, 1], a[:, 2], a[:, 3]
        self.u = np.maximum(a[:, 4], 0.0)
        self.s = np.maximum.accumulate(a[:, 8])       # progress from the line, m
        self.L = float(self.s[-1])
        self.r = _smooth(np.gradient(self.psi, t))
        self.ax = _smooth(np.gradient(self.u, t))
        self.v_mean = self.L / max(self.T, 1e-6)
        self.spec = cars.get(key)
        self.paint = (tuple(int(c) for c in paint[:3]) if paint is not None else None)
        self.t0 = 0.0
        #  the renderer's rim spin state, per car: (phase, last render time, speeds)
        self.spin = (np.zeros(4), None, np.zeros(4))
        #  ... and its wing soup (render's one-car `_mesh3` cache, per car)
        self.wings = dict(wings or {})
        #  ... and the wings' travel through the lap: out in the corners only
        #  (the lateral acceleration smoothed round the lap, across the line)
        self.dep, self.side, self.top = auto_deploy(
            t, self.u * _smooth_lap(np.gradient(self.psi, t)))
        self.mesh = (None, None)
        self.st = self.aux = self.ctl = None
        self.x_now = self.y_now = self.psi_now = self.s_now = 0.0
        self.at(0.0)

    def tau(self, t: float) -> float:
        """Where in its lap the car is at scene time t (s from the line)."""
        return (float(t) + self.t0) % self.T

    def t_at_s(self, s: float) -> float:
        """The lap time at which it passes progress `s` (m, wrapped)."""
        return float(np.interp(float(s) % self.L, self.s, self.t))

    def at(self, t: float) -> None:
        """The car at scene time t: its pose, and the state / HUD / controls
        objects the renderer's car drawing reads."""
        q = self.tau(t)
        tt = self.t
        x, y, psi = (float(np.interp(q, tt, v)) for v in (self.x, self.y, self.psi))
        u, r, ax = (float(np.interp(q, tt, v)) for v in (self.u, self.r, self.ax))
        self.x_now, self.y_now, self.psi_now = x, y, psi
        self.s_now = float(np.interp(q, tt, self.s))
        ay = u * r
        phi = min(max(ROLL_PER_G * ay / 9.81, -0.12), 0.12)   # leans out of the turn
        L = float(getattr(self.spec, "L", 2.5) or 2.5)
        d = math.atan(L * r / max(u, 3.0))                  # the kinematic steer angle
        self.st = SimpleNamespace(x=x, y=y, psi=psi, u=u, v=0.0, r=r, ax=ax, ay=ay, phi=phi)
        w = self.wings
        if w:
            #  the one-panel law (render.flank_deps): the latched side's
            #  OUTER panel at `dep`; the per-flank fields cleared, so the
            #  wing soup's cache key (side, travel) follows the corners
            i = min(int(np.searchsorted(tt, q, side="right")) - 1, len(tt) - 1)
            step = 1.0 / DEP_STEPS
            dep = round(float(np.interp(q, tt, self.dep)) / step) * step
            top = round(float(np.interp(q, tt, self.top)) / step) * step
            w = dict(w, wing_deploy_l=None, wing_deploy_r=None,
                     wing_side=int(self.side[max(i, 0)]),
                     wing_deploy=dep if (w.get("dev_left") or w.get("dev_right")) else 0.0,
                     top_deploy=top if w.get("top_on") else 0.0)
        self.aux = rnd.HudData(V=u, gear=3, delta_wheel=np.array([d, d, 0.0, 0.0]), **w)
        self.ctl = SimpleNamespace(brake=1.0 if ax < BRAKE_AX else 0.0, throttle=0.0)


class Scene:
    """The cars on one map, the scene clock and the camera's car. `advance`
    moves the clock and cuts to the next car every CAM_SWITCH_S s (True at a
    cut); `respace` lays the train out ahead of the camera's car."""

    def __init__(self, pick: dict, paint=None, seed=None, wings=None):
        self.track = pick["track"]
        self.cars = [ReplayCar(k, e, a, paint(k) if paint is not None else None,
                               wings(k) if wings is not None else None)
                     for k, e, a in pick["cars"]]
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.order = [int(i) for i in self.rng.permutation(len(self.cars))]
        self.k = 0
        self.follow = self.cars[self.order[0]]
        self.follow.t0 = float(self.rng.uniform(0.0, self.follow.T))   # anywhere on the lap
        self.t_cut = 0.0
        self.cuts = 0
        self.respace()
        self.update()

    def respace(self) -> None:
        """The other cars ahead of the camera's car, in a random order, each
        GAP_M (+ the closing allowance) ahead of the one behind it."""
        f = self.follow
        s = float(np.interp(f.tau(self.t), f.t, f.s))
        behind = f
        others = [c for c in self.cars if c is not f]
        for i in self.rng.permutation(len(others)):
            c = others[int(i)]
            gap = float(self.rng.uniform(*GAP_M))
            gap += max(0.0, behind.v_mean - c.v_mean) * (CAM_SWITCH_S + FADE_S) * GAP_MARGIN
            s += gap
            c.t0 = (c.t_at_s(s) - self.t) % c.T
            behind = c

    def advance(self, dt: float) -> bool:
        self.t += max(float(dt), 0.0)
        cut = False
        if len(self.cars) > 1 and self.t - self.t_cut >= CAM_SWITCH_S:
            self.k = (self.k + 1) % len(self.order)
            self.follow = self.cars[self.order[self.k]]
            self.t_cut = self.t
            self.cuts += 1
            self.respace()
            cut = True
        self.update()
        return cut

    def update(self) -> None:
        for c in self.cars:
            c.at(self.t)


class _SceneRenderer(rnd.Renderer):
    """`render.Renderer` whose car pass draws EVERY scene car in view, each in
    its own body and paint (the module docstring says why). `drawn` lists the
    last frame's cars as (key, body style, paint) for the self-check."""

    scene = None

    def __init__(self, cfg, track, headless: bool = False):
        super().__init__(cfg, track, headless=headless)
        self.drawn = []

    def _in_view(self, sc) -> list:
        """The scene's cars worth drawing, far to near: the camera's car
        always; another when it is in front of the eye, nearer than FAR_M and
        within half a screen of the frame."""
        c3 = self._cam3
        cs = sc.cars
        P = np.array([(c.x_now, c.y_now) for c in cs], dtype=np.float64)
        dep = c3.ground_depth(P)
        S = c3.ground(P)
        keep = []
        for c, d, px in zip(cs, dep.tolist(), S.tolist()):
            if c is sc.follow or (2.0 * rnd.CHASE_Z_NEAR + 1.0 < d < FAR_M
                                  and -0.5 * self.W < px[0] < 1.5 * self.W):
                keep.append((d, c))
        keep.sort(key=lambda e: -e[0])
        return [c for _, c in keep]

    def _draw_car3d(self, x, y, psi, aux):
        sc = self.scene
        if sc is None or self._cam3 is None:
            return super()._draw_car3d(x, y, psi, aux)
        was = (rnd._CAR, rnd._PAINT, self._st, self._ctl,
               self._spin, self._spin_t, self._spin_w, self._wing3_key, self._wing3)
        drawn = []
        try:
            for c in self._in_view(sc):
                rnd.set_car(c.spec)
                rnd.set_paint(c.paint)
                self._st, self._ctl = c.st, c.ctl
                self._spin, self._spin_t, self._spin_w = c.spin
                self._wing3_key, self._wing3 = c.mesh
                super()._draw_car3d(c.x_now, c.y_now, c.psi_now, c.aux)
                c.spin = (self._spin, self._spin_t, self._spin_w)
                c.mesh = (self._wing3_key, self._wing3)
                drawn.append((c.key, rnd.car_geom().style, rnd._PAINT))
        finally:
            rnd.set_car(was[0])
            rnd.set_paint(was[1])
            (self._st, self._ctl, self._spin, self._spin_t, self._spin_w,
             self._wing3_key, self._wing3) = was[2:]
            self.drawn = drawn


# ==================================================================== #
#  THE SCREEN                                                          #
# ==================================================================== #
def _title_keys() -> dict:
    """The pause menu's keys (`input.MENU_KEYS`) with the title's one
    difference: P, the pause key, is nothing -- there is nothing to pause
    (task 44 review: P closed the game). ESC is 'menu', as a right click and
    a pad's OPTIONS are: it moves the cursor to Quit and never leaves the
    game by itself (task 45: ESC closed the game at once; the footer says
    "ESC to Quit"). Only the window's close quits at once."""
    from .input import MENU_KEYS
    keys = dict(MENU_KEYS)
    keys[pygame.K_ESCAPE] = "menu"
    keys.pop(pygame.K_p, None)
    return keys


def _menu_input(pad=None):
    """The pause menu's input stack in menu mode: the keyboard (and the
    mouse) plus a pad -- the one handed in, else one found now; a pad plugged
    in later is attached by BlendedInput's hot-plug, as while driving. The
    keyboard reads the title's own keys (`_title_keys`)."""
    from .input import BlendedInput, GamepadInput, KeyboardInput
    if pad is None:
        try:
            if GamepadInput.available():
                pad = GamepadInput(0)
        except Exception as exc:          # noqa: BLE001 -- the keyboard still works
            print(f"title: gamepad found but not usable ({exc})")
            pad = None
    if pad is not None:
        pad.seed_edges()                  # the press that ended the last screen is not one here
    kb = KeyboardInput()
    kb.menu_keys = _title_keys()
    inp = BlendedInput(kb, pad, announce=False)
    inp.set_menu(True)
    return inp


_ART: dict = {}


def _art(path: str):
    """The PNG at `path`, loaded once (converted when a window is up)."""
    s = _ART.get(path)
    if s is None:
        s = pygame.image.load(path)
        if pygame.display.get_surface() is not None:
            s = s.convert_alpha()
        s = _ART[path] = s
    return s


def _fit(art, w: int):
    """(art `w` px wide, its black silhouette at 140 alpha: the drop shadow)."""
    w = max(8, int(w))
    s = pygame.transform.smoothscale(art, (w, max(1, round(art.get_height() * w / art.get_width()))))
    shadow = s.copy()
    shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MIN)
    shadow.set_alpha(140)
    return s, shadow


def logo_surfaces(u: float, ss: int | None = None):
    """(logo, shadow) at scale `u`: the owner's logo art (LOGO_ART) LOGO_W x u
    wide, its rule included, and its silhouette for the drop shadow. `ss` is
    kept for the callers (the art is drawn already). Raises if the art will
    not load."""
    return _fit(_art(LOGO_ART), LOGO_W * u)


def subtitle_surfaces(w: int):
    """(subtitle, shadow): the owner's RACE YOUR WING art (SUB_ART) `w` px
    wide. Raises if it will not load."""
    return _fit(_art(SUB_ART), w)


class Title:
    """The title screen. `run()` -> one of ACTIONS. `handle(cmd)` is the
    menu's command vocabulary (drive/menu.py's), `frame(dt)` builds one frame
    (the scene, then the overlay), `pad` is the pad the input stack ends with
    (hand it to the next screen).

    `size` the window; `paint(car) -> rgb | None` a car's paint (None: its
    factory colour); `graphics` the Graphics setting; `bottom` the bottom
    line, [(label, value), ...] or a str (the car, the map and the build the
    first session drives);
    `seed` the scene (None: a new one every launch); `live=False` or no
    reference lap: the panorama; `trace(key)` a reference-lap source (tests);
    `inp` an input stack (tests), else the menu's own."""

    def __init__(self, size=(1280, 800), pad=None, fps: int = 60, graphics: str = "full",
                 paint=None, bottom=(), seed=None, headless: bool = False,
                 live: bool = True, trace=None, inp=None, lib=None):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        self.W, self.H = int(size[0]), int(size[1])
        self.u = min(self.W / 1280.0, self.H / 800.0)
        self.fps = int(fps or 60)
        self.headless = bool(headless)
        #  (label, value) pairs, or one plain line
        self.bottom = ([("", bottom)] if isinstance(bottom, str) else
                       [(str(a), str(b)) for a, b in (bottom or ())])
        self.idx = 0
        self._rows = []                   # the menu rows as last drawn: (i, rect)
        self._armed = None                # the row a mouse press armed
        self._draws = 0                   # frames drawn (the click guard, menu.py's)
        self._fonts = {}
        self._txt_cache = {}
        self._surfs = {}
        self._t = 0.0                     # the screen's clock
        self._fade0 = 0.0                 # the last cut (or the start): fade in from it
        self.ms = 0.0                     # the last frame's build time
        self.fallback_why = ""
        self._car0, self._paint0 = rnd._CAR, rnd._PAINT   # put back by close()
        self.scene = None
        self.r = None
        self._pano = None
        self.mode = "pano"
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption(CAPTION)
        self.inp = inp if inp is not None else _menu_input(pad)
        #  the logo and the menu on the bare background at once: the scene
        #  takes ~1 s to build (the world, the props, the panorama)
        self.screen.fill(C_BG)
        self.draw_overlay(self.screen)
        if not self.headless:
            pygame.display.flip()
        pick = None
        if live:
            try:
                pick = pick_scene(seed, trace=trace)
                if pick is None:
                    self.fallback_why = "no reference lap on any circuit"
                else:
                    self._build_live(pick, paint, seed, graphics, lib)
            except Exception as exc:      # noqa: BLE001 -- a title never stops a launch
                self.fallback_why = f"{type(exc).__name__}: {exc}"
                self.r = self.scene = None
        else:
            self.fallback_why = "live scene off"
        if self.mode != "live":
            self._to_pano((pick or {}).get("track"))
        print("title: " + (f"{self.scene.track}, {len(self.scene.cars)} car(s): "
                           + ", ".join(c.key for c in self.scene.cars)
                           if self.mode == "live" else f"panorama ({self.fallback_why})"))

    # ---- building ------------------------------------------------------
    def _build_live(self, pick, paint, seed, graphics, lib=None) -> None:
        from . import track as trk
        if lib is None:                   # the launch's own library, else the default one
            try:
                from .garage import library
                lib = library()
            except Exception as exc:      # noqa: BLE001 -- then the cars go bare
                print(f"title: no wing library ({type(exc).__name__}: {exc})")
        self.scene = Scene(pick, paint=paint, seed=seed,
                           wings=(lambda k: wing_hud(k, lib)) if lib is not None else None)
        tr = trk.make_track(self.scene.track, surfaces=(SURFACE != "none"))
        cfg = rnd.ViewConfig(size=(self.W, self.H), fps=self.fps, mode="chase", hud="off",
                             **rnd.look_config(graphics))
        cfg.show_vectors = cfg.show_gg = cfg.show_skid = False
        #  every car's painted mesh now (~3-5 ms each), not on the frame it
        #  first comes into view; the renderer prebuilds the camera car's
        for c in self.scene.cars:
            rnd.set_car(c.spec)
            rnd.set_paint(c.paint)
            rnd.car_mesh_cached(rnd.car_geom())
        f = self.scene.follow
        rnd.set_car(f.spec)
        rnd.set_paint(f.paint)
        self.r = _SceneRenderer(cfg, tr, headless=self.headless)
        #  ... and every car's wings as they are now (each car's soup is then
        #  rebuilt at a new travel step only: DEP_STEPS)
        for c in self.scene.cars:
            rnd.set_car(c.spec)
            self.r._wing3_key = self.r._wing3 = None
            self.r._mesh3(c.aux)
            c.mesh = (self.r._wing3_key, self.r._wing3)
        rnd.set_car(f.spec)
        self.r._wing3_key = self.r._wing3 = None
        self.r.scene = self.scene
        self.screen = self.r.screen
        self.mode = "live"

    def _to_pano(self, track=None) -> None:
        """The fallback: a panorama of `track` (the arena's without one)."""
        self.mode = "pano"
        if self.r is not None:
            self.r.scene = None
        self.r = None
        try:
            from . import world as wd
            fl = rnd.Chase3D(self.W, self.H).fl
            self._pano = wd.build_panorama(track or "arena", "circuit", fl)
            self._grass = tuple(wd.GRASS_RGB)
        except Exception as exc:          # noqa: BLE001 -- then the plain background
            print(f"title: no panorama ({type(exc).__name__}: {exc})")
            self._pano = None

    # ---- input -----------------------------------------------------------
    @property
    def pad(self):
        return getattr(self.inp, "pad", None)

    def action(self) -> str:
        return ACTIONS[self.idx]

    def hit(self, cmd_or_xy):
        """The menu row under a point ('hover:X:Y', 'click:X:Y' or (x, y)) as
        last drawn; None off every row."""
        try:
            if isinstance(cmd_or_xy, str):
                _, xs, ys = cmd_or_xy.split(":")
                x, y = float(xs), float(ys)
            else:
                x, y = float(cmd_or_xy[0]), float(cmd_or_xy[1])
        except (ValueError, TypeError):
            return None
        for i, (rx, ry, rw, rh) in self._rows:
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return i
        return None

    def row_centre(self, i: int):
        rx, ry, rw, rh = dict(self._rows)[i]
        return int(rx + rw // 2), int(ry + rh // 2)

    def handle(self, cmd: str):
        """A menu command -> the action to run, or None. The pause menu's
        vocabulary (drive/menu.py): nav_up / nav_down move, select runs the
        row, hover / click / release are the mouse (a press arms a row, the
        release on it runs it; none in the first CLICK_GUARD_DRAWS frames);
        'quit' (the window's close) quits; 'menu' (ESC, a right click,
        OPTIONS) and 'back' (CIRCLE) move the cursor to Quit and never quit
        by themselves -- a stray right click, ESC or a pad's back button must
        not close the game (task 44 review, task 45); ENTER / CROSS then
        quits; 'garage' (BACKSPACE, the touchpad) is the garage, as it is
        everywhere else."""
        n = len(ITEMS)
        if cmd == "quit":
            return "quit"
        if cmd in ("menu", "back"):
            self.idx = ACTIONS.index("quit")
            return None
        if cmd == "garage":
            return "garage"
        if cmd == "nav_up":
            self.idx = (self.idx - 1) % n
        elif cmd == "nav_down":
            self.idx = (self.idx + 1) % n
        elif cmd == "select":
            return self.action()
        elif cmd.startswith(("hover:", "click:", "release:")):
            i = self.hit(cmd)
            if cmd.startswith("release:"):
                armed, self._armed = self._armed, None
                if i is None or i != armed:
                    return None
                self.idx = i
                return self.action()
            if i is None:
                return None
            self.idx = i
            if cmd.startswith("click:"):
                self._armed = i if self._draws >= CLICK_GUARD_DRAWS else None
        return None

    # ---- the frame -------------------------------------------------------
    def frame(self, dt: float) -> float:
        """Advance the scene by dt and build one frame (no flip). Returns its
        build time, ms."""
        t0 = time.perf_counter()
        dt = min(max(float(dt), 0.0), 0.1)
        self._t += dt
        if self.mode == "live":
            try:
                self._draw_live(dt)
            except Exception as exc:      # noqa: BLE001 -- the panorama instead
                print(f"title: the live scene failed ({type(exc).__name__}: {exc}); "
                      f"a panorama instead")
                self.fallback_why = f"{type(exc).__name__}: {exc}"
                self._to_pano(self.scene.track if self.scene is not None else None)
        if self.mode != "live":
            self._draw_pano()
        self._draw_fade()                 # the scene fades in; the menu never blinks
        self.draw_overlay(self.screen)
        self.ms = (time.perf_counter() - t0) * 1e3
        return self.ms

    def _draw_live(self, dt: float) -> None:
        sc, r = self.scene, self.r
        if sc.advance(dt):
            r._chase_live = False         # a cut: the camera snaps to its new car
            r._cam_init = False
            self._fade0 = self._t
        f = sc.follow
        rnd.set_car(f.spec)               # the chase camera frames THIS car
        rnd.set_paint(f.paint)
        r.update_camera(f.st, dt)
        r.draw_frame(f.st, ctl=f.ctl, aux=f.aux)

    def _draw_pano(self) -> None:
        sc, W, H = self.screen, self.W, self.H
        p = self._pano
        if p is None:
            sc.fill(C_BG)
            return
        hy = int(H * HORIZON_FRAC)
        Pw, R = p["Pw"], p["rows"]
        top = hy - R
        if top > 0:
            sc.fill(p["top"], (0, 0, W, top))
        sy0 = max(0, -top)
        hgt = min(R - sy0, hy - max(top, 0))
        if hgt > 0:
            x0 = int(self._t * PAN_PX_S * self.u) % Pw
            w1 = min(W, Pw - x0)
            dy = max(top, 0)
            sc.blit(p["surf"], (0, dy), (x0, sy0, w1, hgt))
            if w1 < W:
                sc.blit(p["surf"], (w1, dy), (0, sy0, W - w1, hgt))
        sc.fill(getattr(self, "_grass", (84, 124, 60)), (0, hy, W, H - hy))

    def _draw_fade(self) -> None:
        a = 1.0 - (self._t - self._fade0) / FADE_S
        if a <= 0.0:
            return
        fs = self._surfs.get("fade")
        if fs is None:
            fs = self._surfs["fade"] = pygame.Surface((self.W, self.H))
            fs.fill((0, 0, 0))
        fs.set_alpha(int(255 * min(a, 1.0)))
        self.screen.blit(fs, (0, 0))

    # ---- text --------------------------------------------------------------
    def _font(self, size: float, bold: bool = False):
        key = (max(6, int(round(size * self.u))), bool(bold))
        f = self._fonts.get(key)
        if f is None:
            for name in FONT_NAMES:
                try:
                    f = pygame.font.SysFont(name, key[0], bold=key[1])
                    if f is not None:
                        break
                except Exception:         # noqa: BLE001
                    continue
            if f is None:
                f = pygame.font.Font(None, key[0] + 4)
            self._fonts[key] = f
        return f

    def _logo(self):
        """(logo, shadow): `logo_surfaces` at this window's scale, built once;
        the name in the menu's font if it cannot be drawn."""
        got = self._surfs.get("logo")
        if got is None:
            try:
                got = logo_surfaces(self.u)
            except Exception as exc:      # noqa: BLE001 -- a logo never stops a launch
                print(f"title: a plain logo ({type(exc).__name__}: {exc})")
                shadow = self._txt(LOGO, 104, (0, 0, 0), True).copy()
                shadow.set_alpha(150)
                got = (self._txt(LOGO, 104, C_LOGO_BOT, True), shadow)
            self._surfs["logo"] = got
        return got

    def _subtitle(self):
        """(subtitle, shadow): `subtitle_surfaces` as wide as the logo's rule,
        built once; SUBTITLE tracked out in the menu's font if it has no art."""
        got = self._surfs.get("subtitle")
        if got is None:
            try:
                got = subtitle_surfaces(self._logo()[0].get_width() * (LOGO_RULE[1] - LOGO_RULE[0]))
            except Exception as exc:      # noqa: BLE001
                print(f"title: a plain subtitle ({type(exc).__name__}: {exc})")
                f = self._font(SUB_SIZE, True)
                gap = max(1, int(round(5 * self.u)))
                gl = [f.render(c, True, C_TEXT) for c in SUBTITLE]
                s = pygame.Surface((sum(g.get_width() for g in gl) + gap * (len(gl) - 1),
                                    max(g.get_height() for g in gl)), pygame.SRCALPHA)
                x = 0
                for g in gl:
                    s.blit(g, (x, 0), special_flags=pygame.BLEND_RGBA_MAX)
                    x += g.get_width() + gap
                s = s.subsurface(s.get_bounding_rect()).copy()
                shadow = s.copy()
                shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MIN)
                shadow.set_alpha(140)
                got = (s, shadow)
            self._surfs["subtitle"] = got
        return got

    def _txt(self, s: str, size: float, col, bold: bool = False):
        key = (s, size, col, bold)
        surf = self._txt_cache.get(key)
        if surf is None:
            if len(self._txt_cache) > 256:
                self._txt_cache.clear()
            surf = self._font(size, bold).render(s, True, col)
            self._txt_cache[key] = surf
        return surf

    def _blit(self, s, x, y, size, col=C_TEXT, bold=False, right=False) -> pygame.Rect:
        surf = self._txt(s, size, col, bold)
        if right:
            x -= surf.get_width()
        return self.screen.blit(surf, (int(x), int(y)))

    # ---- the overlay -------------------------------------------------------
    def _shades(self):
        """The two gradients the text reads on, built once per size: dark
        over the left (full to the menu column's edge, gone TEXT_PX further
        on) and over the bottom lines."""
        got = self._surfs.get("shade")
        if got is not None:
            return got
        W, H, u = self.W, self.H, self.u
        x0, x1 = int(470 * u), int(min(W, 470 * u + 360 * u))
        left = pygame.Surface((max(x1, 1), H), pygame.SRCALPHA)
        xs = np.arange(max(x1, 1), dtype=np.float64)
        k = np.clip((xs - x0) / max(x1 - x0, 1), 0.0, 1.0)
        a = 205.0 * (1.0 - k * k * (3.0 - 2.0 * k))
        alpha = pygame.surfarray.pixels_alpha(left)
        alpha[...] = a[:, None].astype(np.uint8)
        del alpha
        rgb = pygame.surfarray.pixels3d(left)
        rgb[...] = np.array(C_SHADE, dtype=np.uint8)
        del rgb
        bh = int(150 * u)
        low = pygame.Surface((W, max(bh, 1)), pygame.SRCALPHA)
        ys = np.arange(max(bh, 1), dtype=np.float64) / max(bh - 1, 1)
        alpha = pygame.surfarray.pixels_alpha(low)
        alpha[...] = (185.0 * ys ** 1.5)[None, :].astype(np.uint8)
        del alpha
        rgb = pygame.surfarray.pixels3d(low)
        rgb[...] = np.array(C_SHADE, dtype=np.uint8)
        del rgb
        sel = pygame.Surface((int(380 * u), int(46 * u)), pygame.SRCALPHA)
        pygame.draw.rect(sel, (*C_SEL_BG, 215), sel.get_rect(),
                         border_radius=max(3, int(round(8 * u))))
        got = self._surfs["shade"] = (left, low, sel)
        return got

    def layout(self) -> dict:
        """Where everything goes at this size (px): the self-check reads it."""
        u, H = self.u, self.H
        return dict(logo=(int(64 * u), int(58 * u)), menu_y=int(292 * u), row_h=int(52 * u),
                    menu_x=int(48 * u), menu_w=int(380 * u), bottom_y=int(H - 76 * u),
                    footer_y=int(H - 40 * u))

    def draw_overlay(self, screen) -> None:
        """The logo, the subtitle, the menu column, the highlighted row's
        hint, the bottom line and the key hints, on their gradients."""
        self.screen = screen
        u, W, H = self.u, self.W, self.H
        lay = self.layout()
        left, low, sel = self._shades()
        screen.blit(left, (0, 0))
        screen.blit(low, (0, H - low.get_height()))
        # the logo (`_logo`, built once: the owner's art, its rule under the
        # letters) and under it the subtitle, lined up with that rule, each
        # over its drop shadow
        lx, ly = lay["logo"]
        logo, shadow = self._logo()
        off = max(2, int(round(4 * u)))
        screen.blit(shadow, (lx + off, ly + off))
        screen.blit(logo, (lx, ly))
        sub, sub_shadow = self._subtitle()
        sx, sy = lx + int(logo.get_width() * LOGO_RULE[0]), ly + logo.get_height() + int(18 * u)
        screen.blit(sub_shadow, (sx + off, sy + off))
        screen.blit(sub, (sx, sy))
        # the menu column
        mx, my, rh, mw = lay["menu_x"], lay["menu_y"], lay["row_h"], lay["menu_w"]
        self._rows = []
        self._draws += 1
        for i, (label, _a) in enumerate(ITEMS):
            y = my + i * rh
            on = (i == self.idx)
            self._rows.append((i, (mx, y, mw, int(46 * u))))
            if on:
                screen.blit(sel, (mx, y))
                pygame.draw.rect(screen, C_ACCENT, (mx, y + int(8 * u), max(3, int(5 * u)),
                                                    int(30 * u)))
            f = self._font(30, bold=on)
            ty = y + (int(46 * u) - f.get_linesize()) // 2
            self._blit(label, mx + int(26 * u), ty, 30, C_TEXT if on else C_DIM, bold=on)
        hy = my + len(ITEMS) * rh + int(8 * u)
        self._blit(HINTS.get(self.action(), ""), mx + int(26 * u), hy, 16, C_KEY)
        # the bottom line: what Drive starts; the keys; the scene's car
        x = lx
        for label, value in self.bottom:
            if label:
                x += self._blit(label, x, lay["bottom_y"] + int(3 * u), 13, C_KEY).w + int(8 * u)
            x += self._blit(value, x, lay["bottom_y"], 17, C_TEXT).w + int(26 * u)
        self._blit(FOOTER, lx, lay["footer_y"], 14, C_DIM)
        cap = self.caption()
        if cap:
            self._blit(cap, W - int(40 * u), lay["footer_y"], 14, C_DIM, right=True)

    def caption(self) -> str:
        """The live scene's car and map, bottom right ('' for the panorama)."""
        if self.mode != "live" or self.scene is None:
            return ""
        import cars
        from . import track as trk
        f = self.scene.follow
        return (f"{cars.car_name(f.key)}  -  reference lap  -  "
                f"{trk.TRACK_TITLES.get(self.scene.track, self.scene.track)}")

    # ---- the loop ------------------------------------------------------------
    def run(self, max_frames: int | None = None, script=None) -> str:
        """Until an action. `script(frame) -> [commands]` replaces the input
        stack's events (tests); `max_frames` stops a test with 'drive'."""
        clock = pygame.time.Clock()
        clock.tick()
        slow = []
        n = 0
        act = None
        try:
            while act is None:
                dt = clock.tick_busy_loop(self.fps) / 1000.0
                cmds = script(n) if script is not None else self.inp.poll_events()
                for c in cmds or ():
                    act = self.handle(c)
                    if act is not None:
                        break
                if act is not None:
                    break
                ms = self.frame(dt)
                if not self.headless:
                    pygame.display.flip()
                    if self.mode == "live" and n >= 5:     # (the first frames build fonts)
                        slow.append(ms)
                        if len(slow) >= SLOW_FRAMES:
                            med = float(np.median(slow))
                            slow = []
                            if med > SLOW_MS:
                                print(f"title: {med:.0f} ms a frame; a panorama instead "
                                      f"of the live scene")
                                self.fallback_why = f"too slow ({med:.0f} ms a frame)"
                                self._to_pano(self.scene.track if self.scene else None)
                n += 1
                if max_frames is not None and n >= max_frames:
                    act = "drive"
        finally:
            self.close()
        print(f"title -> {act}")
        return act

    def close(self) -> None:
        """Hand the window back: the input out of menu mode (the next screen
        re-seeds the pad's edges), the renderer's module state (the car and
        paint the title pointed it at) as it was, the scene freed."""
        try:
            self.inp.set_menu(False)
        except Exception:                 # noqa: BLE001
            pass
        rnd.set_car(self._car0)
        rnd.set_paint(self._paint0)
        if self.r is not None:
            self.r.scene = None
        self.r = None
        self.scene = None
        # the Renderer froze itself into gc's permanent generation; the
        # session's own unfreezes on construction, the garage's does not
        gc.unfreeze()
        gc.collect()


# ==================================================================== #
#  SELF-CHECK                                                          #
# ==================================================================== #
def self_check(verbose: bool = True, shots: str | None = None) -> bool:
    """python3 -m drive.title [--shots DIR]: the layout at three sizes, every
    action, the mouse, the scene's picks over many seeds, each car in its own
    body and paint, the train, the fallback and the frame budget. `shots`
    saves the frames drawn as PNGs there."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import cars
    from . import track as trk
    ok = True
    n_row = [0]

    def rep(tag, passed, msg=""):
        nonlocal ok
        ok = ok and bool(passed)
        n_row[0] += 1
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {n_row[0]:2d} {tag}"
                  + (f": {msg}" if msg else ""))

    def shot(t_, name):
        if shots:
            os.makedirs(shots, exist_ok=True)
            pygame.image.save(t_.screen, os.path.join(shots, name))

    class _NoInput:                       # the tests feed commands to handle()
        pad = None

        def poll_events(self):
            return []

        def set_menu(self, flag):
            pass

        def seed_edges(self):             # ... and stands in for a pad (row 11):
            pass                          # no real pad reaches a self-check

    import contextlib
    import io
    quiet = contextlib.redirect_stdout(io.StringIO())
    # 1-3 the layout at three sizes: inside the window, nothing overlapping,
    # the left side darkened under the text, the scene live
    for size in ((1280, 800), (1920, 1080), (960, 600)):
        with quiet:
            t = Title(size, seed=7, headless=True, inp=_NoInput(),
                      bottom=[("CAR", "Opel Corsa C 1.2"), ("MAP", "Linden park"),
                              ("BUILD", "my corsa")])
        for _ in range(40):               # past the fade in
            t.frame(1.0 / 60.0)
        shot(t, f"title_{size[0]}x{size[1]}.png")
        W, H = size
        rows = [r_ for _, r_ in t._rows]
        lay = t.layout()
        inside = all(0 <= x and 0 <= y and x + w <= W and y + h <= H for x, y, w, h in rows)
        menu_bottom = max(y + h for _, y, _, h in rows)
        foot_top = lay["bottom_y"]
        logo_bottom = (lay["logo"][1] + t._logo()[0].get_height() + int(18 * t.u)
                       + t._subtitle()[0].get_height())    # the subtitle's foot
        clear = menu_bottom + int(40 * t.u) <= foot_top and logo_bottom < lay["menu_y"]
        #  the bottom line fits; the key hints and the longest caption share a row
        lx, f14 = lay["logo"][0], t._font(14)
        worst = max((f"{cars.car_name(k)}  -  reference lap  -  {trk.TRACK_TITLES[m]}"
                     for k in cars.CAR_ORDER for m in trk.CIRCUITS), key=lambda s_: f14.size(s_)[0])
        line = sum(t._font(13).size(a_)[0] + t._font(17).size(b_)[0] + int(34 * t.u)
                   for a_, b_ in t.bottom)
        wide = (lx + line <= W
                and lx + f14.size(FOOTER)[0] + int(24 * t.u) <= W - int(40 * t.u) - f14.size(worst)[0])
        #  under the shade, right of the menu's rows and left of where it fades
        a = pygame.surfarray.pixels3d(t.screen)
        lum_l = float(a[int(436 * t.u):int(466 * t.u), int(300 * t.u):int(600 * t.u)].mean())
        del a
        rep(f"layout {W}x{H}: six rows inside the window, clear of the logo and the bottom "
            f"line; the menu reads on a dark side; the scene is live",
            inside and clear and wide and lum_l < 90.0 and t.mode == "live",
            f"menu {rows[0][1]}..{menu_bottom} px, bottom line at {foot_top}, "
            f"left-side luminance {lum_l:.0f}, {t.caption()}")
        t.close()
    # 4 every action, by the keys: rows in order, ENTER runs the highlighted one
    got = []
    for i in range(len(ITEMS)):
        t.idx = 0
        for _ in range(i):
            t.handle("nav_down")
        got.append(t.handle("select"))
    t.idx = 0
    t.handle("nav_up")
    wrap = t.action()
    esc = []
    for c in ("menu", "back", "quit"):
        t.idx = 0
        esc.append((t.handle(c), t.action()))
    rep("the keys: UP / DOWN move (wrapping), ENTER runs the row -- Drive, Challenges, "
        "Garage, Tutorial, Settings, Quit; 'quit' (the window's close) quits; "
        "'menu' / 'back' (ESC, a right click, OPTIONS, CIRCLE) only move to Quit; "
        "BACKSPACE is the garage",
        got == list(ACTIONS) and wrap == "quit"
        and esc == [(None, "quit"), (None, "quit"), ("quit", "drive")]
        and t.handle("garage") == "garage" and t.handle("nav_left") is None
        and t.handle("reset") is None, f"{got}, up from Drive -> {wrap}, {esc}")
    # 5 the mouse: hover moves the cursor, press + release on one row runs it,
    # nothing in the guard frames after the screen opens, a drag off the row
    with quiet:
        t = Title((1280, 800), seed=3, headless=True, inp=_NoInput(), live=False)
    t.frame(0.0)
    x2, y2 = t.row_centre(2)
    x4, y4 = t.row_centre(4)
    early = (t.handle(f"click:{x2}:{y2}"), t.handle(f"release:{x2}:{y2}"))
    for _ in range(CLICK_GUARD_DRAWS):
        t.frame(0.0)
    t.handle(f"hover:{x4}:{y4}")
    hov = t.action()
    miss = (t.handle("click:5:5"), t.handle("release:5:5"))
    drag = (t.handle(f"click:{x2}:{y2}"), t.handle(f"release:{x4}:{y4}"))
    click = (t.handle(f"click:{x2}:{y2}"), t.handle(f"release:{x2}:{y2}"))
    rep("the mouse: hover highlights, a click on a row runs it; not in the first "
        f"{CLICK_GUARD_DRAWS} frames, not off a row, not dragged off it",
        early == (None, None) and hov == "settings" and miss == (None, None)
        and drag == (None, None) and click == (None, "garage"),
        f"early {early}, hover -> {hov}, miss {miss}, drag {drag}, click {click}")
    t.close()
    # 6 the picks over many seeds: a circuit, 1..5 cars, distinct types, every
    # number and every type seen, each car with its own lap on that map
    picks = [pick_scene(s_) for s_ in range(300)]
    ns = [len(p["cars"]) for p in picks]
    maps_seen = sorted({p["track"] for p in picks})
    types = sorted({k for p in picks for k, _e, _a in p["cars"]})
    distinct = all(len({k for k, _e, _a in p["cars"]}) == len(p["cars"]) for p in picks)
    own = all(a.shape[1] == 9 and len(a) > 100 for p in picks for _k, _e, a in p["cars"])
    rep("300 seeds: a dressed circuit, 1 to 5 cars, never two of one type; every number "
        "of cars and every type seen",
        all(p["track"] in trk.CIRCUITS for p in picks) and min(ns) == 1
        and max(ns) == len(cars.CAR_ORDER) and set(ns) == set(range(1, 6)) and distinct
        and types == sorted(cars.CAR_ORDER) and own and set(maps_seen) == set(trk.CIRCUITS),
        f"cars {dict((k, ns.count(k)) for k in range(1, 6))}, maps {maps_seen}")
    # 7 a live scene of all five: every car in view drawn in its OWN body and
    # paint, the render module's car and paint put back after each frame
    pal = {"corsa": (40, 72, 186), "mx5": None, "540i": (226, 226, 220),
           "express": (30, 92, 56), "bus": (112, 26, 44)}
    s5 = next(s_ for s_ in range(300) if len(pick_scene(s_)["cars"]) == 5)
    car0, paint0 = rnd._CAR, rnd._PAINT
    with quiet:
        t = Title((1280, 800), seed=s5, headless=True, inp=_NoInput(), paint=pal.get)
    seen, styles_ok, multi = {}, True, 0
    from .bodies import style_of
    for i in range(int(3.2 * CAM_SWITCH_S * 30)):
        t.frame(1.0 / 30.0)
        d = t.r.drawn
        multi = max(multi, len(d))
        for key, style, paint in d:
            seen[key] = (style, paint)
            styles_ok = styles_ok and style == style_of(cars.get(key)) and paint == pal[key]
        if i in (20, 330, 640):
            shot(t, f"title_live_{i:03d}.png")
    restored = (t.r._st is not None and rnd._PAINT == t.scene.follow.paint
                and rnd._CAR is t.scene.follow.spec)
    #  the owner's "deployed with all the wings" -- "only in the curve"
    #  (2026-09-27): both flanks and the top wing on every car, in its own
    #  wing soup (more polygons than its bare body); through a lap each car's
    #  outer panel and top wing are out in the corners (|ay| well past AY_ON:
    #  mostly at full travel, on the outer flank) and stowed on the straights (|ay| < 1
    #  and not just out of a corner), and the frames' aux say so
    bare = {}
    for c in t.scene.cars:
        rnd.set_car(c.spec)
        bare[c.key] = len(rnd.car_mesh_cached(rnd.car_geom()).starts)
    rnd.set_car(t.scene.follow.spec)

    def _lap(c):
        ay = c.u * c.r
        n_ = len(ay)
        #  a straight sample: no corner within TOP_HOLD + a ramp either side
        near = np.convolve((np.abs(ay) > AY_OFF).astype(float),
                           np.ones(2 * int(1.5 / max(np.median(np.diff(c.t)), 1e-3)) + 1),
                           mode="same") > 0
        straight = (np.abs(ay) < 1.0) & ~near
        corner = np.abs(ay) > AY_ON + 2.0
        outer = np.sign(ay) == c.side
        return dict(
            #  (not every mid-corner sample: the panel takes t_ext to come
            #  out, and a chicane's swap waits for it to stow first -- as in
            #  the game; measured 0.69..0.94 of them at full travel)
            corner_out=bool(corner.any() and np.mean(c.dep[corner] > 0.99) > 0.6
                            and np.mean(c.top[corner] > 0.99) > 0.8
                            and np.mean(outer[corner]) > 0.8),
            #  (a lap with no straight at all -- the arena's, some cars -- has
            #  nothing to stow on: vacuous; `straights` below wants some car's)
            straight_in=bool(not straight.any() or (c.dep[straight].max() < 0.01
                                                    and c.top[straight].max() < 0.01)),
            straights=int(straight.sum()), n=n_)
    laps = {c.key: _lap(c) for c in t.scene.cars}
    moved = {c.key: set() for c in t.scene.cars}
    for i in range(int(12.0 * 30)):
        t.frame(1.0 / 30.0)
        for c in t.scene.cars:
            moved[c.key].add((c.aux.wing_deploy > 0.99, c.aux.wing_deploy < 0.01))
    wing_rows = {c.key: (bool(c.aux.dev_left and c.aux.dev_right and c.aux.top_on),
                         (c.aux.wing_deploy_l, c.aux.wing_deploy_r),
                         0 if c.mesh[1] is None else len(c.mesh[1].starts) - bare[c.key])
                 for c in t.scene.cars}
    wings_ok = all(a and d == (None, None) and n > 0 for a, d, n in wing_rows.values())
    auto_ok = (all(v["corner_out"] and v["straight_in"] for v in laps.values())
               and any(v["straights"] > 0 for v in laps.values()))
    live_ok = sum(1 for m in moved.values() if (True, False) in m and (False, True) in m) >= 3
    t.close()
    back = rnd._CAR is car0 and rnd._PAINT == paint0
    rep("five cars: each drawn in its own body and paint (styles "
        + ", ".join(sorted({s_ for s_, _p in seen.values()})) + "); several in one frame; "
        "the camera's car and paint left set between frames, the launch's back on close",
        styles_ok and len(seen) >= 3 and multi >= 2 and restored and back,
        f"drawn {sorted(seen)}, at most {multi} in a frame")
    rep("the wings: every car with both side wings and the top wing "
        f"({WINGS_CONFIG} set), drawn by the one-panel law", wings_ok,
        ", ".join(f"{k} +{n} polys" for k, (_a, _d, n) in wing_rows.items()))
    rep("the wings deploy in the corners only: the outer panel and the top wing out "
        "mid-corner, stowed on the straights, on every car's lap; out and in on screen",
        auto_ok and live_ok,
        ", ".join(f"{k} {'ok' if v['corner_out'] and v['straight_in'] else v}"
                  for k, v in laps.items())
        + f"; out and in in 12 s: {sum(1 for m in moved.values() if len(m) > 1)} cars")
    # 8 the train and the camera: a cut every CAM_SWITCH_S s to another car;
    # no two cars within 12 m of each other along the lap over the shots
    with quiet:
        sc = Scene(pick_scene(s5), seed=11)
    follows, closest, cuts = [sc.follow.key], 1e9, 0

    def _ahead():                         # right after a (re)spacing: the train is ahead
        return all(((c.s_now - sc.follow.s_now) % c.L) >= GAP_M[0] - 1.0
                   for c in sc.cars if c is not sc.follow)
    ahead = _ahead()
    for i in range(int(5.5 * CAM_SWITCH_S * 20)):
        if sc.advance(0.05):
            cuts += 1
            follows.append(sc.follow.key)
            ahead = ahead and _ahead()
        ss = sorted(c.s_now for c in sc.cars)
        L = sc.cars[0].L
        gaps = np.diff(ss + [ss[0] + L])
        closest = min(closest, float(gaps.min()))
    rep("the chase camera cuts to the next car every 10 s (each car in turn); the train "
        "stays spaced -- never two cars within 12 m",
        cuts == 5 and len(set(follows[:5])) == 5 and closest > 12.0 and ahead,
        f"{cuts} cuts, cameras {follows}, closest {closest:.1f} m")
    # 9 the fallback: no reference lap anywhere, a renderer that throws, the
    # live scene off -- the panorama (sky over grass), the menu still works
    with quiet:
        tn = Title((1280, 800), headless=True, inp=_NoInput(), trace=lambda k: None)
        for _ in range(8):
            tn.frame(0.1)
    a = pygame.surfarray.pixels3d(tn.screen)
    sky = a[900:1200, 60:200].mean(axis=(0, 1))
    grass = a[900:1200, 560:640].mean(axis=(0, 1))
    del a
    shot(tn, "title_panorama.png")
    none_ok = (tn.mode == "pano" and "no reference" in tn.fallback_why
               and sky[2] > sky[1] and grass[1] > grass[0] and tn.handle("select") == "drive")
    tn.close()
    with quiet:
        tb = Title((1280, 800), seed=5, headless=True, inp=_NoInput())

        def _boom(*_a, **_k):
            raise RuntimeError("boom")
        tb.r.draw_frame = _boom
        tb.frame(1.0 / 60.0)
    boom_ok = tb.mode == "pano" and "boom" in tb.fallback_why and tb.r is None
    tb.close()
    with quiet:
        to = Title((960, 600), headless=True, inp=_NoInput(), live=False)
    off_ok = to.mode == "pano" and to._pano is not None
    to.close()
    rep("the fallback -- no reference lap, a scene that throws, the scene off: a panned "
        "panorama (sky over grass) under the same menu",
        none_ok and boom_ok and off_ok,
        f"sky {tuple(int(v) for v in sky)}, grass {tuple(int(v) for v in grass)}; "
        f"throws -> {tb.mode}; off -> {to.mode}")
    # 10 run(): the input stack's commands reach it; the pad comes back
    with quiet:
        tr_ = Title((960, 600), seed=2, headless=True, inp=_NoInput(), live=False)
        act = tr_.run(script=lambda n: ["nav_down"] * 4 + ["select"] if n == 3 else [])
        tq = Title((960, 600), seed=2, headless=True, inp=_NoInput(), live=False)
        act_q = tq.run(script=lambda n: ["quit"] if n == 2 else [])
    rep("run(): the menu's commands pick Settings; the window's close quits; the input "
        "stack's pad is handed back", act == "settings" and act_q == "quit" and tr_.pad is None,
        f"{act}, {act_q}")
    # 11 the REAL input stack (the keyboard and mouse the launch reads): a
    # right click, P and ESC never close the game from the title -- the menu
    # vocabulary calls them 'menu' (task 44 review; ESC, task 45) -- a right
    # click or ESC moves to Quit and ENTER there quits; P does nothing
    real = {}
    for name, evs in (("right click", [dict(type=pygame.MOUSEBUTTONDOWN, button=3, pos=(5, 5))]),
                      ("P", [dict(type=pygame.KEYDOWN, key=pygame.K_p, mod=0)]),
                      ("right click, ENTER",
                       [dict(type=pygame.MOUSEBUTTONDOWN, button=3, pos=(5, 5)),
                        dict(type=pygame.KEYDOWN, key=pygame.K_RETURN, mod=0)]),
                      ("ESC", [dict(type=pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)]),
                      ("ESC, ENTER", [dict(type=pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0),
                                      dict(type=pygame.KEYDOWN, key=pygame.K_RETURN, mod=0)])):
        with quiet:
            ti = Title((960, 600), seed=2, headless=True, live=False,
                       inp=_menu_input(_NoInput()))
            ti.frame(0.0)
            pygame.event.clear()
            for e_ in evs:
                pygame.event.post(pygame.event.Event(e_.pop("type"), **e_))
            real[name] = (ti.run(max_frames=4), ti.action())
    rep("the real keyboard and mouse: a right click or ESC only moves to Quit (ENTER there "
        "quits), P does nothing; ESC never returns 'quit' by itself",
        real == {"right click": ("drive", "quit"), "P": ("drive", "drive"),
                 "right click, ENTER": ("quit", "quit"), "ESC": ("drive", "quit"),
                 "ESC, ENTER": ("quit", "quit")},
        str(real))
    # 12 the frame budget at 1280x800, five cars, the full look: the scene
    # and the overlay together, load-normalised like every page's
    with quiet:
        tf = Title((1280, 800), seed=s5, headless=True, inp=_NoInput(), paint=pal.get)
    for _ in range(20):
        tf.frame(1.0 / 60.0)
    calib = rnd.cpu_calibration()
    ts = []
    for _ in range(300):
        ts.append(tf.frame(1.0 / 60.0))
    ts = np.array(ts)
    okb, why = rnd.frame_budget_verdict(float(ts.mean()), float(np.percentile(ts, 99)),
                                        budget_mean=12.0, budget_p99=16.0, calib=calib)
    rep(f"frame budget, 1280x800, {len(tf.scene.cars)} cars, full look, the overlay: "
        f"mean <= 12 ms, p99 <= 16 ms", okb, why)
    tf.close()
    if verbose:
        print(f"title self-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    _shots = None
    if "--shots" in sys.argv[1:]:
        i_ = sys.argv.index("--shots")
        _shots = sys.argv[i_ + 1] if i_ + 1 < len(sys.argv) else "runs/title_shots"
    sys.exit(0 if self_check(shots=_shots) else 1)
