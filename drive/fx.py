"""drive/fx.py -- smoke, dust, spray, and the camera jolt.

What this module is
-------------------
The particles a tyre throws off and the camera shake the ground puts into the
chase camera:

  * SMOKE from a tyre that slides or spins on tarmac or a kerb, and ONLY past
    the grip peak (see "When a tyre smokes" below): lateral |alpha| past the
    car's onset (SLIP_ALPHA0_CAR: Corsa 11.0, MX-5 11.7, 540i 13.8 deg), or
    |kappa| past SLIP_KAPPA_HOLD (0.40) over N_LOCK (4) frames and T_LOCK
    (120 ms). A plough past the limit, a rear axle in a slide, a driven
    wheel spinning, a locked wheel: all read off the same two numbers, so
    nothing here has to know which wheels are driven.
  * DUST off the road: on grass a light, dry, brownish veil that thins
    quickly, and cut grass and earth flicked up; on gravel beige dust and
    small stones. Speed makes it, slip makes more; the rear wheels throw
    most of it (the fronts' dust stays under the body).
  * SPRAY on a wet road at speed: a fine light mist behind each tyre and the
    streaks of the drops it flings up.
  * the JOLT: a fine vertical buzz on a kerb, slower bumps with a little
    lateral off the road, nothing on tarmac.

A CONSUMER of HudData, like the sound (drive/audio.py): it reads the slip,
the surface under each wheel and the pose, and never writes anything back.
It does not import `render` at module level (render imports it): the
renderer is handed in as `rnd`, and the one render table it needs
(WHEEL_XY, the car's footprint) is read through sys.modules at call time.

When a tyre smokes -- measured on the real physics, not guessed
----------------------------------------------------------------
In a time-trial game smoke says "you are over the limit", so it must start
PAST the grip peak: a well-driven corner and an ABS stop are clean. Measured
with drive/vehicle.py on every car in cars.CAR_ORDER (the `_rig` ramp steer
at 2.2 deg/s with vehicle.ramp_steer's own peak and abort -- the grip-limit
driver the contract names -- and straight-line stops and launches; runs of
1 ms steps; this file's self-check re-runs the tight cases for each car):
  * lateral, PER CAR (SLIP_ALPHA0_CAR, looked up by HudData.car_key): the
    largest |alpha| on ANY wheel up to the peak lateral acceleration,
    8-40 m/s, wing off, is 10.14-10.71 deg on the Corsa (largest at 40
    m/s), 10.04-11.41 on the MX-5 (40 m/s) and 12.47-13.36 on the 540i (12
    m/s): the heavy car's fronts run 2-3 deg further at their peak, so the
    Corsa's 11 deg smoked the 540i 0.6-0.9 s BEFORE its peak. It is always
    a front (the rears reach 4.5-7.1 deg). Onset = the largest, wing or
    not (next point), + 0.29 deg: 11.0 / 11.7 / 13.8; an unknown car
    (car_key '') gets the Corsa's. Past the peak the fronts plough on
    (+2.5 s: 15.1-19.6 deg) -- that is the slide that smokes, full at
    onset + 7 deg; in the rig its first puff comes 0.11-0.96 s after the
    peak (every car, 8-40 m/s).
  * with the flank panel deployed (VehicleConfig wing 'fin' or 'plate') the
    peak is the tyres' to 22 m/s (Corsa 10.79, MX-5 11.21, 540i 12.89 deg
    with the plate). Faster, the panel's side force keeps a_y climbing
    AFTER the fronts are past their own grip peak -- the front axle's
    utilisation tops out at 9.8-10.6 deg on the Corsa and the MX-5 and
    12.5-13.5 on the 540i, wing or not -- until the car departs: Corsa fin
    at 30 / 32 / 33.75 m/s peaks with the fronts at 11.9 / 12.9 / 15.3 deg
    and beta 6-9 deg, and from 34 m/s ramp_steer aborts (|beta| > 12 deg).
    That front angle has no ceiling short of the departure, and those
    tyres ARE sliding, so a panel's run counts up to the EARLIER of its
    peak a_y and the fronts' grip peak (13.51 deg the most, the 540i plate
    at 40 m/s). So from 24-27 m/s (the 540i: 38-40 m/s with the plate) a
    wing-carried corner smokes a little before its peak a_y (Corsa fin 32
    m/s: 8 puffs by it, 0 by the fronts' grip peak): the tyres are past
    their limit and the wing is holding the car on.
  * the axle utilisation is NOT a usable gate: it peaks at 0.92-0.99 at
    the limit (never 1.0) and FALLS as the tyre slides further (0.97 1 s
    past the peak, 0.84-0.87 8 s past it at 29 deg), and a handbrake-locked
    axle reads 0.14. So it is only a soft 0.5 -> 0.8 gate that stops a
    wheel with next to no lateral load (one in the air) from smoking.
  * longitudinal: ABS cycling at full pedal, stopped down to the 2 m/s
    smoke speed gate, peaks at kappa 0.36 / 0.49 / 0.54 on dry tarmac
    (Corsa / MX-5 / 540i, from 8-45 m/s; the excursions grow as the car
    slows) and 0.74 at mu 0.63, while a lock-up sits at kappa ~1 as long
    as the pedal is down and a clutch dump spins at 0.8-1.5. No level
    clears ABS by depth alone; DURATION does: past 0.40 the longest ABS
    excursion lasts 31 ms dry and 84 ms at mu 0.63 (540i from 15 m/s),
    a lock-up 0.96-0.98 s of a 1 s stop, a launch 0.17-2.0 s. But timing
    one run is not enough: the span from the first to the latest frame
    over a level can bridge TWO excursions when the frames miss the dip
    between them. The old hold (0.18 for 60 ms, span only) leaked 1-5
    puffs a stop at 24-30 fps with frame jitter; replaying 60 stops (3
    cars, mu 1.0 / 0.8 / 0.63 and the grass's 0.55, from 8-45 m/s) at 14
    frame rates from 20 to 240 fps with 0 / +-25 / +-50 % jitter (14280
    runs), it let smoke through in 3370 of the 10710 on the road. So
    |kappa| must stay past SLIP_KAPPA_HOLD (0.40) over N_LOCK (4) frames in
    a row AND for T_LOCK (120 ms); then the wheel smokes until |kappa|
    falls back under SLIP_KAPPA0 (0.18), where the excess ramp starts, so
    the smoke fades out instead of cutting off. The same 14280 runs: 0
    smoke-capable frames (and 0 puffs in 1140 replays through
    Effects.update itself, 24-144 fps); a lock-up puffs from 0.14-0.26 s,
    a clutch dump with TC off from 0.14-0.59 s. On the road a mu below
    0.95 reads 'wet' (scenery.wheel_surfaces), where nothing smokes at
    all -- the low-mu cases are the margin.
  * a launch spins its wheels at 2-6 m/s of car speed, so the smoke's speed
    gate is the faster of the car and the spinning tyre's own surface,
    V * (1 + kappa), from 2 m/s.

The cost model -- measured on THIS machine (pygame 2.5.2, SDL 2.28.3, dummy
driver, 1280x800), and it decides the drawing strategy
---------------------------------------------------------------------------
  * a per-pixel-alpha sprite blits at 0.6 ns/px (15 us for a 162 px
    square) -- pygame's SIMD blender. The SAME sprite with `set_alpha`
    modulating it falls off that path: x4.2. So a particle's alpha is
    BAKED: the sprite cache is keyed on (kind, shape variant, size bucket,
    alpha level, haze level, shading) and every blit is a plain one.
  * `convert_alpha` + RLEACCEL is SLOWER here (15.6 against 2.9 us at R 32):
    re-encoding the RLE per alpha change costs more than it skips. Not used.
  * a sprite built from scratch (numpy colour ramp + alpha) is 119 us at
    162 px; an alpha level DERIVED from the full-alpha sprite of the same
    key (copy, then the profile times a constant straight into its alpha
    plane) is 19 us; the lumpy alpha profile under both is 488 us, so all
    72 profiles are built with the Effects (~10 ms), never in a frame. Only
    the full-alpha "base" is built from scratch, and new sprites are capped
    per frame by count and by estimated build time (MAKE_MAX, MAKE_US_MAX;
    one always allowed so the cache fills); past that a puff borrows the
    nearest cached alpha level or size, or sits out a frame. The cache is
    least-recently-used and bounded in PIXELS (CACHE_PX_MAX, 24 MB at 800
    rows): it used to grow to 1316 sprites / 32 MB and then clear() in one
    2.7 ms hitch. Over a mixed session (5 surfaces x 3 zooms) the worst fx
    frame fell from 3.7-4.5 ms to 1.8-1.9 ms, and the plan view's from 1.3
    to 0.8 ms.
  * on-screen size is capped at r_cap (R_CAP_PX, 80 px at 800 rows): a puff
    3 m from the eye is physically 300 px across. Beyond the cap a puff
    FADES instead of growing (alpha * (cap / r)^CAP_FADE), and it fades out
    entirely between NEAR_CULL and NEAR_CULL + NEAR_FADE of depth. A
    per-frame pixel budget drops the faintest puffs past it, with hysteresis
    both ways (last frame's survivors rank BUDGET_HYST brighter, last
    frame's cut ones as much fainter): a plain cut made 13-47 puffs a slide
    blink off for one frame; now 0 (self-check, both resolutions).
  * full-pool cost, update + both passes, four-wheel slide (load ~6):
    chase 0.74-0.89 ms at 1280x800 and 0.94-1.1 ms at 1920x1080 (blits are
    ~70 %: 0.55 ms at 0.8 M px), car_up 0.47-0.49 ms; a wet road at 24-28
    m/s 0.67-0.69 / 0.89-0.92 ms; idle 0.02 ms.
  * cut grass, stones and water drops are STREAKS, not sprites: a line from
    where the particle was STREAK_T ago (relative to the camera) to where it
    is, alpha-blended by pygame.gfxdraw (0.9 us a line). A 3 px round sprite
    read as a fly hanging beside the car; a streak reads as debris in flight.

Resolution
----------
The chase lens's focal length is proportional to the window height, so a
puff is 1.35x wider on screen at 1080 rows than at 800 -- but a cap and a
budget fixed in pixels were not: at 1920x1080 the alpha-weighted smoke
coverage fell from 0.129 to 0.044 of the screen (polka dots). So the caps
scale with k = H / 800. The budget cannot simply scale with k^2: in a
four-wheel slide the pool is full and the budget binds at both sizes, so
the cost IS the budget (1.42 M px, 1.18 ms at 1080). What buys it back is
alpha per pixel -- coverage per blitted pixel is the mean alpha of what is
drawn -- so at k > 1 the puffs come FEWER and DENSER: rate x k^-0.6,
alpha x k^1.4, budget x k^0.75 (0.99 M px at 1080). Measured, same run,
full pool: coverage x1.03 of 800 rows at 0.98-1.00 ms (alpha x k^1.2 and
budget x k: x1.01 at 1.02-1.05 ms; k^2 alone: x0.90 at 1.17-1.18 ms).
LARGER puffs (size x k^0.15-0.3) did not help: the extra area hits the cap
and fades. Budgets: 1.0 ms at 800 rows, 1.2 ms at 1080.

What the chase view can show, and the numbers that follow from it
------------------------------------------------------------------
The eye rides 6.4-7 m behind the CG and the bottom of the frame meets the
road 4.3 m from it, so a puff born at a rear wheel (5.4 m deep) is at most
~3 m of travel from leaving the frame. Air-still smoke would cross that in
0.2 s at 16 m/s. Particles therefore leave with a FRACTION of the car's
velocity (`inh`: the car's wake entrains them) and are thrown outward and
up, so a slide reads as smoke billowing at the rear corners and the sides
rather than a flicker at the bottom edge. The near/far split is the car
footprint's nearest corner, not the CG (see `_split_depth`).

The camera jolt -- the chase view's; the plan views have task 27's
---------------------------------------------------------------------
The chase camera is a camera on a car: a kerb buzzes it and grass bumps it.
The plan views are a MAP. At the plan's 6-14 px/m a 3 cm jolt is 0.2-0.4 px:
invisible as motion, and what it does do is flip the rounding of every thin
edge line between two pixels, which reads as the whole map shimmering. So
`jolt()` is (0, 0) in the plan views; there the shake is task 27's own
(render.shake_offset: whole pixels, at most SHAKE_PX, the anchor moves).
Both are gated by the harness's shake strength (`update_camera(shake=)`:
the Shake setting, the pause, the wheels, the speed) -- the renderer applies
this jolt only while that strength is above 0 -- and by `ViewConfig.shake`.

Determinism
-----------
No live RNG and no wall clock in anything that decides what is drawn. The
particle draws come from a PCG64 generator seeded from the track name
(zlib.crc32), re-seeded by `reset()`; the jolt is integer-hash value noise
of a phase accumulated from the render clock's deltas. Same frames in ->
bit-identical pool arrays out (the self-check asserts it). The wind (a
constant 1 m/s drift, direction from the same crc) is a layout constant.
`time.perf_counter` is used ONLY for the `ms_*` instrumentation.

Run `python3 -m drive.fx` for the self-check.
"""
from __future__ import annotations

import math
import sys
import time
import zlib
from collections import OrderedDict

import numpy as np
import pygame
import pygame.gfxdraw as _gfx

from . import world as _world

# ======================================================================= #
#  CONSTANTS                                                              #
# ======================================================================= #
# --- the pool ------------------------------------------------------------
POOL_N = 160              # particles, 'high' detail. Four wheels at full slip
                          # saturate it, and then the oldest, most faded
                          # puffs are recycled first
POOL_N_LOW = 80           # 'low' detail
EMIT_FRAME_MAX = 10       # new particles per frame at most (600/s at 60 fps)
DT_MAX = 0.05             # s  a frame longer than this ages the pool by 0.05 s
                          #    (a hitch must not teleport the smoke)

# --- when a tyre smokes: PAST the grip peak (module docstring, measured) ---
#: |alpha| past which a wheel smokes, PER CAR (HudData.car_key): the largest
#: |alpha| on any wheel up to the ramp-steer rig's peak a_y, 8-40 m/s, wing
#: off / fin / plate (a panel's run counted up to the fronts' own grip peak
#: when that comes first), plus 0.29 deg
SLIP_ALPHA0_CAR = {
    'corsa': math.radians(11.0),     # 10.71 (40 m/s, wing off)
    'mx5': math.radians(11.7),       # 11.41 (40 m/s, wing off)
    '540i': math.radians(13.8),      # 13.51 (40 m/s, plate; 13.36 wing off)
}
SLIP_ALPHA0 = SLIP_ALPHA0_CAR['corsa']   # an unknown car: the Corsa's
SLIP_ALPHA_SPAN = math.radians(7.0)  # full at onset + 7 deg (Corsa: 18)
SLIP_UTIL0 = 0.50                    # soft gate on the axle's utilisation:
SLIP_UTIL1 = 0.80                    #   none below 0.5, full from 0.8
SLIP_KAPPA0 = 0.18                   # |kappa| slip: the excess counts from here
SLIP_KAPPA_HOLD = 0.40               # ... once |kappa| has been past THIS
N_LOCK = 4                           #   over this many frames in a row
T_LOCK = 0.120                       #   AND this long, s (ABS: <= 84 ms on
                                     #   the road); then until < SLIP_KAPPA0
SLIP_KAPPA_SPAN = 0.30               # full at 0.48
V_WALK = 1.5              # m/s  walking pace: nothing is emitted below it
V_SMOKE_MIN = 2.0         # m/s  of max(car, tyre surface): no smoke below
V_SMOKE_FULL = 15.0       # m/s  the smoke rate's speed factor saturates here
V_DUST_FULL = 15.5        # m/s
V_SPRAY0 = 6.0            # m/s  below this a wet tyre drips, it does not mist
V_SPRAY_FULL = 30.0       # m/s

# --- the kinds -------------------------------------------------------------
K_SMOKE, K_DUST, K_BITS, K_GDUST, K_STONE, K_SPRAY, K_DROP = range(7)
KIND_NAMES = ('smoke', 'dust', 'grass', 'gravel', 'stone', 'spray', 'drop')
N_KINDS = 7
#: a soft puff (a sprite) or a streak (a line: debris and drops)
_PUFFY = (True, True, False, True, False, True, False)

#: Per-kind parameters, all est (tuned by eye against the screenshots, then
#: sized so the pool holds a slide). Columns:
#:   rate  particles/s per wheel at full intensity
#:   life  s (lo, hi)          r0 / r1  m, radius at birth / at death (lo, hi)
#:   a     peak alpha 0..1     tau  s, drag relaxation toward the wind
#:   rise  m/s buoyant rise the drag relaxes toward; grav m/s^2 (the bits)
#:   inh   fraction of the car's velocity a particle leaves with
#:   up    m/s initial vertical (lo, hi); out m/s outward from the car (lo, hi)
#:   jit   m/s random horizontal; back m behind the contact point; z0 m
#:   fin   s fade-in; fout  exponent of the (1-u) fade-out
#:   dil   dilution exponent: alpha * (r0/r)^dil as a puff spreads
#: Grass dust lives 0.8-1.3 s and dilutes at ^0.55 (was 1.0-1.6 s, ^0.3):
#: dense and pale where the tyre throws it, gone a car length behind, not a
#: cloud that hangs round the car. Spray is many small faint puffs (r 0.14 -> 1.2 m, alpha 0.34)
#: plus the drops' streaks, not four grey cumulus puffs.
_KP = {
    #          rate  life        r0            r1          a     tau   rise  grav
    K_SMOKE: (40.0, (1.6, 2.6), (0.45, 0.60), (2.6, 3.6), 0.85, 0.70, 1.00, 0.0,
              # inh  up          out         jit  back  z0    fin   fout  dil
              0.45, (1.6, 2.8), (1.0, 2.6), 0.9, 0.25, 0.25, 0.06, 1.2, 0.30),
    K_DUST:  (22.0, (0.8, 1.3), (0.30, 0.45), (1.5, 2.2), 0.95, 0.45, 0.35, 0.0,
              0.40, (0.9, 1.6), (0.6, 1.6), 0.8, 0.35, 0.15, 0.05, 1.3, 0.40),
    K_BITS:  (9.0, (0.5, 0.9), (0.020, 0.035), (0.020, 0.035), 0.95, 1.2, 0.0, 9.81,
              0.45, (1.2, 2.8), (0.5, 2.0), 1.0, 0.30, 0.10, 0.02, 0.35, 0.0),
    K_GDUST: (24.0, (1.0, 1.6), (0.36, 0.48), (1.7, 2.5), 0.66, 0.55, 0.35, 0.0,
              0.35, (0.9, 1.6), (0.6, 1.6), 0.8, 0.35, 0.15, 0.06, 1.4, 0.45),
    K_STONE: (9.0, (0.5, 0.9), (0.015, 0.030), (0.015, 0.030), 0.95, 2.0, 0.0, 9.81,
              0.30, (1.2, 3.0), (1.0, 3.0), 1.2, 0.30, 0.08, 0.02, 0.35, 0.0),
    K_SPRAY: (32.0, (0.5, 0.9), (0.14, 0.22), (0.9, 1.3), 0.34, 0.35, 0.10, 0.0,
              0.62, (0.6, 1.4), (0.3, 1.0), 0.6, 0.35, 0.12, 0.04, 1.1, 0.40),
    K_DROP:  (26.0, (0.25, 0.45), (0.01, 0.01), (0.01, 0.01), 0.45, 0.8, 0.0, 9.81,
              0.55, (1.4, 3.0), (0.3, 1.1), 0.5, 0.30, 0.06, 0.02, 0.6, 0.0),
}

#: lit (top) and shaded (underside) colour per kind; a streak uses the lit
#: one. Est, and deliberately NOT the HUD's signal colours (C_YELLOW /
#: C_GREEN / C_BAR_BRK / C_PURPLE / the PB ghost's green), which the render
#: self-checks count. The dust colours are set against world.GRASS_RGB
#: (84, 124, 60) and GRAVEL_RGB (184, 170, 136): the old (196, 184, 140) /
#: (140, 130, 98) ramp was the grass's own luminance (Y 129 against 120),
#: so over the car's contact shadow it read as a dark olive smudge. Dust is
#: lit and thin: lighter (Y 204) and warm, with a shallow shade.
_LIT = np.array([
    (236, 236, 238),      # smoke: near-white
    (228, 210, 170),      # grass dust: light, dry, brownish
    (78, 84, 42),         # cut grass and earth: darker than the verge
    (232, 218, 186),      # gravel dust: pale beige, lighter than the trap
    (132, 124, 108),      # stones: darker than the trap they came from
    (232, 237, 243),      # spray: light blue-white
    (214, 222, 232),      # drops
], dtype=np.float64)
_SHADE = np.array([
    (170, 173, 182),      # the underside takes the sky's blue
    (204, 186, 148),
    (60, 68, 34),
    (204, 190, 160),
    (96, 90, 82),
    (206, 214, 226),
    (190, 200, 214),
], dtype=np.float64)

# --- drawing ---------------------------------------------------------------
RES_REF_H = 800.0         # px  the window height every px constant is set at
R_MIN_PX = 1.2            # px  smallest sprite radius
R_CAP_PX = 80.0           # px  largest ('high') at 800 rows; see the docstring
R_CAP_PX_LOW = 56.0       # px  ('low')
CAP_FADE = 1.3            # -   past the cap: alpha * (cap / r)^CAP_FADE. At
                          #     0.6 the capped puffs stayed dense and, drawn
                          #     smaller than they are, stopped overlapping: a
                          #     row of polka dots at the bottom of the frame
PX_BUDGET = 0.8e6         # px  blitted per frame at most ('high') at 800 rows,
PX_BUDGET_LOW = 0.4e6     #     ~0.45 ms at 0.55 ns/px; the faintest go first
BUDGET_HYST = 0.30        # -   last frame's survivors rank this much brighter,
                          #     last frame's cut ones this much fainter
HIRES_RATE = -0.6         # -   k = H/800 > 1: puffy emission x k^HIRES_RATE,
HIRES_ALPHA = 1.4         #     alpha x k^HIRES_ALPHA, budget x k^2 x
HIRES_BUDGET = -1.25      #     k^HIRES_BUDGET (module docstring, Resolution)
N_BUCKETS = 24            # size buckets, geometric R_MIN..cap: ~20 % apart
N_ALEV = 16               # baked alpha levels
N_HAZE = 5                # haze levels 0 .. world.HAZE_MAX
N_VAR = 3                 # shape variants of a puff (the lumps' angles)
NEAR_CULL = 1.0           # m  camera depth: nearer than this is not drawn
NEAR_FADE = 2.0           # m  ... and it fades in over this much depth
STREAK_T = 0.030          # s  a streak runs from where the particle was this
                          #    long ago (relative to the camera) to where it is
STREAK_WIDE_DEPTH = 4.0   # m  nearer than this a streak is 2 px wide
STREAK_CULL = 2.0         # m  streaks fade out between here and +STREAK_FADE:
STREAK_FADE = 2.0         #    a drop passing the lens is a long diagonal
                          #    line from the vanishing point -- rain, not spray
STREAK_MAX_PX = 22.0      # px (at 800 rows) longest streak on screen
MAKE_MAX = 14             # new sprites per frame at most ...
MAKE_US_MAX = 300.0       # us ... and at most this much estimated build time
                          #    (one is always allowed). Build cost measured:
MAKE_US_BASE = (20.0, 0.004)      # us + us/px from scratch (119 us at 162 px)
MAKE_US_DERIVED = (7.0, 0.0005)   # us + us/px derived from the base (19 us)
                          #    The lumpy alpha PROFILE under them is 27 us +
                          #    18 ns/px (488 us at 162 px): all 3 x 24 are
                          #    built with the Effects (~7 ms at 800 rows),
                          #    never in a frame
CACHE_PX_MAX = 6.0e6      # px of sprites kept (24 MB at 800 rows; x (H/800)^2)
CACHE_MAX = 2000          # sprites at most; least recently used go first

# --- the camera jolt (est; chase only, see the module docstring) -----------
JOLT_KERB_M = (0.010, 0.025)    # m  kerb buzz amplitude at V_JOLT_LO .. _HI
JOLT_KERB_HZ = (16.0, 24.0)     # Hz ... its rate (a ridged kerb at speed)
JOLT_ROUGH_M = (0.030, 0.060)   # m  grass / gravel bumps
JOLT_GRASS_HZ = (6.0, 8.0)      # Hz
JOLT_GRAVEL_HZ = (8.0, 10.0)    # Hz gravel is a finer, harsher texture
JOLT_LAT = 0.35                 # -  lateral / vertical, off the road only
V_JOLT_LO = 3.0                 # m/s
V_JOLT_HI = 25.0                # m/s
JOLT_ATTACK = 0.04              # s  onto the kerb: at once
JOLT_RELEASE = 0.16             # s  off it: a short ring-down
JOLT_SNAP = 1e-4                # an envelope below this is exactly zero

WIND_MS = 1.0             # m/s  the drift the smoke relaxes toward (est: a
                          #      light summer breeze; direction from the crc)


# ======================================================================= #
#  HELPERS                                                                #
# ======================================================================= #
def _hash01(n: int, seed: int) -> float:
    """Integer hash -> [-1, 1). Platform-exact: no float sin() hashing."""
    h = (n * 0x9E3779B1 + seed * 0x85EBCA77 + 0x165667B1) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 2147483648.0 - 1.0


def _vnoise(ph: float, seed: int) -> float:
    """Smooth value noise in [-1, 1]: one hashed value per integer phase,
    smoothstep-blended. One `ph` unit is one bump."""
    i = math.floor(ph)
    f = ph - i
    f = f * f * (3.0 - 2.0 * f)
    a = _hash01(int(i), seed)
    b = _hash01(int(i) + 1, seed)
    return a + (b - a) * f


def _clip01(v: float) -> float:
    return 0.0 if v <= 0.0 else (1.0 if v >= 1.0 else v)


def _f(v, default: float = 0.0) -> float:
    """float(v), or `default` for None, junk, NaN or inf. HudData always
    carries floats; this is so a partial aux (a test, a menu frame) can never
    raise out of the renderer."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _vec4(v):
    """Four finite floats from a per-wheel field, or None when it is absent
    or not four long. A bad entry reads as 0 (no slip)."""
    if v is None:
        return None
    try:
        if len(v) != 4:
            return None
        return [_f(v[i], 0.0) for i in range(4)]
    except TypeError:
        return None


def _render_mod():
    """drive.render, read at call time (render imports this module)."""
    m = sys.modules.get('drive.render')
    if m is None:                               # pragma: no cover
        from . import render as m               # lazy, never at module level
    return m


def _kp(j):
    """Column j of _KP as an array indexed by kind: (N_KINDS,), or
    (N_KINDS, 2) for the (lo, hi) ranges."""
    return np.array([_KP[k][j] for k in range(N_KINDS)], dtype=np.float64)


_RATE, _LIFE, _R0, _R1, _APK, _TAU, _RISE, _GRAV, _INH, _UP, _OUT, _JIT, \
    _BACK, _Z0, _FIN, _FOUT, _DIL = (_kp(j) for j in range(17))
#: the per-particle copy of a kind's constants, one row per particle so an
#: emission sets them with ONE gather (P_* are its columns)
P_TAU, P_RISE, P_GRAV, P_FIN, P_FOUT, P_DIL, P_APK = range(7)
_PBLOCK = np.column_stack([_TAU, _RISE, _GRAV, _FIN, _FOUT, _DIL, _APK])
_PBLOCK_IDLE = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
_STREAK = np.array([not p for p in _PUFFY], dtype=bool)


# ======================================================================= #
#  THE EFFECTS                                                            #
# ======================================================================= #
class Effects:
    """Particles and the camera jolt for one Renderer.

    `update` once per drawn frame (before anything is drawn: the jolt it
    computes moves the eye every layer projects through), then
    `draw(far=True)` under the car and `draw(far=False)` over it.
    """

    def __init__(self, rnd):
        cfg = getattr(rnd, 'cfg', None)
        self.low = getattr(cfg, 'detail', 'high') == 'low'
        self.n = POOL_N_LOW if self.low else POOL_N
        self.emit_scale = 0.5 if self.low else 1.0
        name = getattr(getattr(rnd, 'track', None), 'name', '') or ''
        self.seed = zlib.crc32(('fx:' + str(name)).encode())
        wa = (self.seed % 3600) / 3600.0 * 2.0 * math.pi
        self._wind = np.array([WIND_MS * math.cos(wa), WIND_MS * math.sin(wa)])

        n = self.n
        # the pool: everything a particle is, as parallel arrays
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.age = np.zeros(n)
        self.life = np.ones(n)
        self.r0 = np.zeros(n)
        self.r1 = np.zeros(n)
        self.inten = np.zeros(n)
        self.kind = np.zeros(n, dtype=np.int64)
        self.var = np.zeros(n, dtype=np.int64)
        self.alive = np.zeros(n, dtype=bool)
        # per-particle copies of the kind's constants (vectorised step)
        self.par = np.zeros((n, len(_PBLOCK_IDLE)))
        # derived each frame: the radius (m) and alpha (0..1) to draw
        self.rad = np.zeros(n)
        self.alp = np.zeros(n)

        # sprites -----------------------------------------------------------
        self._cache = OrderedDict()
        self._cache_px = 0
        self._prof = {}
        self._wh = None
        self._setup_res(int(getattr(rnd, 'W', 1280) or 1280),
                        int(getattr(rnd, 'H', 800) or 800))

        # instrumentation (perf_counter: timing only, never layout)
        self.ms_update = 0.0
        self.ms_draw = 0.0
        self.sprites_made = 0
        self.make_skips = 0         # puffs not drawn for a frame (build cap)
        self.budget_drops = 0
        self.blinks = 0             # drawn, budget-cut, drawn again: 0 wanted
        self.stat_cov = 0.0         # sum(alpha * pi r^2) / screen, drawn puffs
        self.stat_px = 0            # px blitted this frame
        self.stat_lines = 0         # streaks drawn this frame
        self.stat_make_us = 0.0     # estimated sprite build time this frame
        self.stat_made = 0          # sprites built this frame
        self.enabled = True
        self.reset()

    # ------------------------------------------------------------------ #
    def _setup_res(self, W: int, H: int) -> None:
        """Everything that is in pixels, for a W x H window (module
        docstring, "Resolution"). Called again if the window changes."""
        self._wh = (W, H)
        k = max(0.5, min(H / RES_REF_H, 3.0))
        self.res_k = k
        hi = max(k, 1.0)
        self.r_cap = (R_CAP_PX_LOW if self.low else R_CAP_PX) * k
        self.px_budget = ((PX_BUDGET_LOW if self.low else PX_BUDGET) * k * k
                          * hi ** HIRES_BUDGET)
        self.cache_px_max = CACHE_PX_MAX * k * k
        # fewer, denser puffs at high resolution (puffy kinds only)
        puffy = np.array(_PUFFY, dtype=bool)
        self._rate_k = np.where(puffy, hi ** HIRES_RATE, 1.0)
        self._pblock = _PBLOCK.copy()
        self._pblock[:, P_APK] *= np.where(puffy, hi ** HIRES_ALPHA, 1.0)
        self._wide = k >= 1.25          # streaks 2 px wide at 1000+ rows
        ratio = (self.r_cap / R_MIN_PX) ** (1.0 / (N_BUCKETS - 1))
        self._inv_log_ratio = 1.0 / math.log(ratio)
        self._rb = R_MIN_PX * ratio ** np.arange(N_BUCKETS)
        self._dim = (2 * np.ceil(self._rb) + 2).astype(np.int64)
        self._ctr = (self._dim - 1) * 0.5          # sprite centre, px
        self._cache.clear()
        self._cache_px = 0
        self._prof = {}
        for b in range(N_BUCKETS):
            for var in range(N_VAR):
                self._profile(var, b)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Empty the pool, re-seed, zero the jolt. Same state as new."""
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        self.alive[:] = False
        for a in (self.pos, self.vel, self.age, self.r0, self.r1, self.inten,
                  self.rad, self.alp):
            a[...] = 0.0
        self.life[:] = 1.0
        self.par[:] = _PBLOCK_IDLE
        self.kind[:] = 0
        self.var[:] = 0
        self._serial = 0
        # emission accumulators, per wheel per kind. A (wheel, kind) that is
        # NOT emitting sits at ONSET, so the first puff comes on the frame
        # the slip starts rather than 1/rate seconds later.
        self._acc = np.full((4, N_KINDS), 0.95)
        # the lock-up / wheelspin hold, per wheel: the run past
        # SLIP_KAPPA_HOLD (frames, s), and whether it has passed (_hold)
        self._k_n = [0] * 4
        self._k_held = [0.0] * 4
        self._k_on = [False] * 4
        self._t_last = None
        self._frame = 0
        self._prep_frame = -1
        self._far_items = []
        self._near_items = []
        self._far_lines = []
        self._near_lines = []
        self._vc = np.zeros(3)
        # the budget's hysteresis and the blink count, pool-indexed
        self._drawn = np.zeros(self.n, dtype=bool)
        self._drawn_pp = np.zeros(self.n, dtype=bool)
        self._cut_p = np.zeros(self.n, dtype=bool)
        # jolt state
        self._env_k = 0.0
        self._env_r = 0.0
        self._ph_k = 0.0
        self._ph_r = 0.0
        self._jolt = (0.0, 0.0)
        self.emitted = np.zeros(N_KINDS, dtype=np.int64)

    # ------------------------------------------------------------------ #
    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def counts(self) -> dict:
        """{kind name: live particles}."""
        k = self.kind[self.alive]
        return {KIND_NAMES[i]: int((k == i).sum()) for i in range(N_KINDS)}

    def state(self) -> tuple:
        """The pool arrays, for the determinism check."""
        return (self.pos, self.vel, self.age, self.life, self.r0, self.r1,
                self.inten, self.kind, self.var, self.alive, self.par,
                self.rad, self.alp)

    # ------------------------------------------------------------------ #
    #  INPUTS                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def surfaces(aux) -> tuple:
        """Per wheel 'tarmac' | 'wet' | 'kerb' | 'grass' | 'gravel'.

        aux.surf4 when the harness filled it (drive/scenery.py
        `wheel_surfaces`); otherwise the physics' own flags: off the track ->
        grass (the sim's surface model has no gravel), mu < 0.95 -> wet (the
        wet and damp patches are 0.63 and ~0.8)."""
        s4 = getattr(aux, 'surf4', None)
        on = bool(getattr(aux, 'on_track', True))
        mu = _vec4(getattr(aux, 'mu', None))
        try:
            s4 = s4 if s4 is not None and len(s4) == 4 else None
        except TypeError:
            s4 = None
        out = []
        for i in range(4):
            s = s4[i] if s4 is not None else None
            if s is None:
                if not on:
                    s = 'grass'
                else:
                    m = 1.0 if mu is None or mu[i] == 0.0 else mu[i]
                    s = 'wet' if m < 0.95 else 'tarmac'
            out.append(str(s))
        return tuple(out)

    @staticmethod
    def alpha0(aux) -> float:
        """The car's lateral smoke onset, rad (SLIP_ALPHA0_CAR): by
        aux.car_key ('corsa' | 'mx5' | '540i'), else a guess from
        aux.car_name the way drive/audio.py's profile_key makes it (the
        session fills only the name today), else the Corsa's."""
        a0 = SLIP_ALPHA0_CAR.get(str(getattr(aux, 'car_key', '') or '').lower())
        if a0 is not None:
            return a0
        name = str(getattr(aux, 'car_name', '') or '').lower()
        if 'mx-5' in name or 'mx5' in name or 'miata' in name:
            return SLIP_ALPHA0_CAR['mx5']
        if '540' in name:
            return SLIP_ALPHA0_CAR['540i']
        return SLIP_ALPHA0

    @classmethod
    def excess(cls, aux, lon_ok=None) -> np.ndarray:
        """(4,) slip excess per wheel, 0 = gripping, 1 = a full slide (<= 1.5).

        Lateral: |alpha| past the car's onset (`alpha0`), the largest slip
        angle any wheel of THAT car reaches up to its peak lateral
        acceleration (module docstring), softly gated by the axle's
        utilisation. Longitudinal: |kappa| past SLIP_KAPPA0, either sign
        (spin or lock) -- counted only for the wheels in `lon_ok` when
        given: the hold `update` keeps (`_hold`), which is what keeps an ABS
        stop clean. None = no hold (the instantaneous excess, for rigs that
        feed single samples)."""
        # four wheels: plain floats, ~4x cheaper than numpy on (4,) arrays
        al = _vec4(getattr(aux, 'alpha', None))
        ka = _vec4(getattr(aux, 'kappa', None))
        uf = _f(getattr(aux, 'util_f', 0.0))
        ur = _f(getattr(aux, 'util_r', 0.0))
        a0 = cls.alpha0(aux)
        du = SLIP_UTIL1 - SLIP_UTIL0
        gf = min(max((uf - SLIP_UTIL0) / du, 0.0), 1.0)
        gr = min(max((ur - SLIP_UTIL0) / du, 0.0), 1.0)
        out = np.zeros(4)
        for i in range(4):
            a = abs(al[i]) if al is not None else 0.0
            k = abs(ka[i]) if ka is not None else 0.0
            e_lat = min(max((a - a0) / SLIP_ALPHA_SPAN, 0.0), 1.5) \
                * (gf if i < 2 else gr)
            e_lon = min(max((k - SLIP_KAPPA0) / SLIP_KAPPA_SPAN, 0.0), 1.5)
            if lon_ok is not None and not lon_ok[i]:
                e_lon = 0.0
            out[i] = e_lat if e_lat > e_lon else e_lon
        return out

    def _hold(self, ka, dt: float) -> list:
        """Per wheel: may its |kappa| smoke? Only once |kappa| has stayed
        past SLIP_KAPPA_HOLD over N_LOCK frames in a row AND for T_LOCK,
        timed from the first of those frames to this one; from then on
        until it falls back under SLIP_KAPPA0, so the smoke fades with the
        excess instead of cutting off at the hold level.

        The span alone is NOT a lower bound on one true run: at 24-30 fps
        with frame jitter the frames can miss the dip between two ABS
        excursions and the span bridges both (1-5 puffs a stop at kappa
        0.18 / 60 ms). The deeper level is what makes that rare -- above
        0.40 an on-road ABS excursion lasts at most 84 ms, against 120 ms
        -- and the frame count keeps two or three long, jittered frames
        from making up T_LOCK on their own (module docstring)."""
        out = []
        for i in range(4):
            k = abs(ka[i]) if ka is not None else 0.0
            if self._k_on[i] and k > SLIP_KAPPA0:
                out.append(True)
                continue
            if k > SLIP_KAPPA_HOLD:
                if self._k_n[i] > 0:
                    self._k_held[i] += dt
                self._k_n[i] += 1
            else:
                self._k_n[i] = 0
                self._k_held[i] = 0.0
            on = (self._k_n[i] >= N_LOCK
                  and self._k_held[i] >= T_LOCK - 1e-9)
            self._k_on[i] = on
            out.append(on)
        return out

    _WXY = None

    @classmethod
    def wheels(cls, aux, x, y, psi) -> np.ndarray:
        """(4,2) world contact points: aux.wheels_xy (task 27's HudData
        field, drive.hud_data fills it from Sim.wheel_world), else the older
        aux.wheel_xy, else pose + WHEEL_XY."""
        w = getattr(aux, 'wheels_xy', None)
        if w is None or (hasattr(w, '__len__') and len(w) == 0):
            w = getattr(aux, 'wheel_xy', None)
        if w is not None:
            try:
                w = np.asarray(w, dtype=np.float64)
                if w.shape == (4, 2) and np.isfinite(w).all():
                    return w
            except (TypeError, ValueError):
                pass
        wx = cls._WXY
        if wx is None:
            wx = cls._WXY = np.asarray(_render_mod().WHEEL_XY, dtype=np.float64)
        c, s = math.cos(psi), math.sin(psi)
        return np.column_stack([x + wx[:, 0] * c - wx[:, 1] * s,
                                y + wx[:, 0] * s + wx[:, 1] * c])

    @staticmethod
    def car_velocity(rnd, aux, psi) -> np.ndarray:
        """(2,) world ground velocity: the state's body (u, v) when it has
        them, else |V| along psi + beta."""
        st = getattr(rnd, '_st', None)
        u = _f(getattr(st, 'u', None), math.nan)
        v = _f(getattr(st, 'v', None), math.nan)
        c, s = math.cos(psi), math.sin(psi)
        if math.isfinite(u) and math.isfinite(v):
            return np.array([u * c - v * s, u * s + v * c])
        V = _f(getattr(aux, 'V', 0.0))
        b = math.radians(_f(getattr(aux, 'beta_deg', 0.0)))
        return np.array([V * math.cos(psi + b), V * math.sin(psi + b)])

    # ------------------------------------------------------------------ #
    #  UPDATE                                                             #
    # ------------------------------------------------------------------ #
    def update(self, rnd, x: float, y: float, psi: float, aux, ctl) -> None:
        """Once per drawn frame, before anything is drawn.

        dt is the render clock's delta (rnd._t_render), clamped to 0..DT_MAX
        and scaled by the slow-motion factor so the smoke slows with the car;
        nothing ages while aux.paused (the pause menu's frame is frozen)."""
        t0 = time.perf_counter()
        self._frame += 1
        t = _f(getattr(rnd, '_t_render', 0.0))
        dt = 0.0 if self._t_last is None else t - self._t_last
        self._t_last = t
        dt = min(max(dt, 0.0), DT_MAX)
        paused = bool(getattr(aux, 'paused', False))
        if paused:
            dt = 0.0
        dt *= min(max(_f(getattr(aux, 'time_scale', 1.0), 1.0), 0.0), 4.0)
        V = _f(getattr(aux, 'V', 0.0))
        surfs = self.surfaces(aux)
        pose_ok = math.isfinite(x) and math.isfinite(y) and math.isfinite(psi)
        # ViewConfig.effects switched off in a running session: the renderer
        # built this object once, so honour the flag here -- empty pool, no
        # emission. The jolt is ViewConfig.shake's, and carries on.
        self.enabled = bool(getattr(getattr(rnd, 'cfg', None), 'effects', True))
        if not self.enabled:
            if self.alive.any():
                self.alive[:] = False
                self.alp[:] = 0.0
            dt_p = 0.0
        else:
            dt_p = dt
        if dt_p > 0.0:
            live = self.alive.any()
            if live:
                self._step(dt)
            ka = _vec4(getattr(aux, 'kappa', None))
            e4 = self.excess(aux, self._hold(ka, dt))
            rates, inten = self._rates(V, surfs, e4, ka)
            emit = rates.any() and pose_ok
            if emit or live:
                # the car's ground velocity: the particles inherit part of
                # it, and the streaks are drawn relative to it (the camera)
                vc = self.car_velocity(rnd, aux, psi) if pose_ok else self._vc[:2]
                self._vc = np.array([vc[0], vc[1], 0.0])
            if emit:
                self._emit(dt, rates, inten, self.wheels(aux, x, y, psi), psi, vc)
            else:
                self._acc[:] = 0.95         # idle = ONSET (see reset)
            if live or emit:
                self._derive()
        self._update_jolt(rnd, V, surfs, dt, paused)
        self.ms_update = (time.perf_counter() - t0) * 1e3
        self.ms_draw = 0.0

    def _step(self, dt: float) -> None:
        """Drag toward the wind (and the buoyant rise), gravity on the bits,
        integrate, age, kill. Vectorised over the whole pool; dead slots are
        stepped too (cheaper than a mask) and never read."""
        if not self.alive.any():
            return
        par = self.par
        rise, grav = par[:, P_RISE], par[:, P_GRAV]
        k = np.exp(-dt / par[:, P_TAU])
        vxy = self.vel[:, :2]
        vxy -= self._wind
        vxy *= k[:, None]
        vxy += self._wind
        vz = self.vel[:, 2]
        vz -= rise
        vz *= k
        vz += rise
        vz -= grav * dt
        self.pos += self.vel * dt
        self.age += dt
        ballistic = grav > 0.0
        dead = (self.age >= self.life) | (ballistic & (self.pos[:, 2] <= 0.0))
        self.alive &= ~dead
        # a puff never sinks through the road
        np.maximum(self.pos[:, 2], np.where(ballistic, -1.0, 0.05),
                   out=self.pos[:, 2])

    def _rates(self, V: float, surfs, e4, ka=None) -> tuple:
        """(4, N_KINDS) particles/s and (4, N_KINDS) intensity 0.5..1.15."""
        rates = np.zeros((4, N_KINDS))
        inten = np.zeros((4, N_KINDS))
        if V <= V_WALK:
            return rates, inten
        fv_d = _clip01((V - V_WALK) / (V_DUST_FULL - V_WALK)) ** 0.8
        fv_w = _clip01((V - V_SPRAY0) / (V_SPRAY_FULL - V_SPRAY0)) ** 0.8
        for i in range(4):
            s, e = surfs[i], float(e4[i])
            e1 = min(e, 1.0)
            ie = 0.5 + 0.5 * e1 + 0.3 * max(e - 1.0, 0.0)
            front = i < 2
            if s in ('tarmac', 'kerb'):
                if e > 0.0:
                    # the rubber's own speed: a spinning tyre at a launch
                    # grinds at V * (1 + kappa) while the car does 2-6 m/s
                    k = ka[i] if ka is not None else 0.0
                    v_rub = V * (1.0 + max(k, 0.0))
                    if v_rub > V_SMOKE_MIN:
                        fv_s = _clip01((v_rub - V_SMOKE_MIN)
                                       / (V_SMOKE_FULL - V_SMOKE_MIN)) ** 0.6
                        rates[i, K_SMOKE] = _RATE[K_SMOKE] * e1 ** 0.7 * fv_s
                        inten[i, K_SMOKE] = ie
            elif s == 'grass':
                # the fronts' dust mostly stays under the body; drawn at the
                # rears' rate it peeked out beside the car as round orbs
                fk = (0.25 + 0.75 * e1) if front else 1.0
                rates[i, K_DUST] = _RATE[K_DUST] * fv_d * (0.8 + 0.6 * e1) * fk
                rates[i, K_BITS] = _RATE[K_BITS] * fv_d * (0.4 + 1.0 * e1)
                inten[i, K_DUST] = (0.55 + 0.45 * fv_d * (0.5 + 0.5 * e1)) \
                    * (0.7 if front else 1.0)
                inten[i, K_BITS] = 1.0
            elif s == 'gravel':
                fk = (0.35 + 0.65 * e1) if front else 1.0
                rates[i, K_GDUST] = _RATE[K_GDUST] * fv_d * (0.7 + 0.8 * e1) * fk
                rates[i, K_STONE] = _RATE[K_STONE] * fv_d * (0.5 + 1.0 * e1)
                inten[i, K_GDUST] = (0.6 + 0.4 * fv_d * (0.5 + 0.5 * e1)) \
                    * (0.75 if front else 1.0)
                inten[i, K_STONE] = 1.0
            elif s == 'wet':
                if V > V_SPRAY0:
                    # the rears run in the fronts' wake: est +25 %
                    wk = fv_w * (1.25 if not front else 1.0)
                    rates[i, K_SPRAY] = _RATE[K_SPRAY] * wk * (1.0 + 0.6 * e1)
                    rates[i, K_DROP] = _RATE[K_DROP] * wk
                    inten[i, K_SPRAY] = 0.55 + 0.45 * fv_w
                    inten[i, K_DROP] = 0.6 + 0.4 * fv_w
        rates *= self._rate_k[None, :] * self.emit_scale
        return rates, inten

    def _emit(self, dt, rates, inten, wxy, psi, vc) -> None:
        on = rates > 0.0
        acc = self._acc
        acc[~on] = 0.95                     # idle = ONSET (see reset)
        acc[on] += rates[on] * dt
        cnt = np.floor(acc)
        acc -= cnt
        total = int(cnt.sum())
        if total <= 0:
            return
        if total > EMIT_FRAME_MAX:          # the budget: scale every stream
            cnt = np.floor(cnt * (EMIT_FRAME_MAX / total))
            total = int(cnt.sum())
            if total <= 0:
                return
        cnt_i = cnt.astype(np.int64).reshape(-1)
        wheel = np.repeat(np.repeat(np.arange(4), N_KINDS), cnt_i)
        kind = np.repeat(np.tile(np.arange(N_KINDS), 4), cnt_i)
        ie = inten.reshape(-1)[np.repeat(np.arange(4 * N_KINDS), cnt_i)]
        n = total

        # slots: free ones first, then the most-faded live ones
        free = np.flatnonzero(~self.alive)
        if free.size >= n:
            slots = free[:n]
        else:
            live = np.flatnonzero(self.alive)
            frac = self.age[live] / self.life[live]
            steal = live[np.argsort(-frac, kind='stable')[:n - free.size]]
            slots = np.concatenate([free, steal])

        U = self._rng.random((n, 10))
        c, s = math.cos(psi), math.sin(psi)
        h = np.array([c, s])
        nr = np.array([-s, c])
        side = np.where(wheel % 2 == 0, 1.0, -1.0)      # FL RL left (+y)
        # front-wheel dust: thrown less far out (it stays in the body's
        # silhouette) and carried back faster, into the rears' trail
        fdust = (wheel < 2) & ((kind == K_DUST) | (kind == K_GDUST))

        base = (wxy[wheel] - _BACK[kind][:, None] * h
                + (side * 0.06 + (U[:, 7] - 0.5) * 0.14)[:, None] * nr
                + ((U[:, 9] - 0.5) * 0.20)[:, None] * h)
        z0 = _Z0[kind] + 0.10 * U[:, 7]
        lo, hi = _LIFE[kind, 0], _LIFE[kind, 1]
        life = lo + (hi - lo) * U[:, 0]
        sz = 0.85 + 0.25 * np.clip(ie, 0.0, 1.2)     # denser = a bigger plume
        r0 = (_R0[kind, 0] + (_R0[kind, 1] - _R0[kind, 0]) * U[:, 1]) * sz
        r1 = (_R1[kind, 0] + (_R1[kind, 1] - _R1[kind, 0]) * U[:, 2]) * sz
        up = _UP[kind, 0] + (_UP[kind, 1] - _UP[kind, 0]) * U[:, 3]
        out = _OUT[kind, 0] + (_OUT[kind, 1] - _OUT[kind, 0]) * U[:, 4]
        out = np.where(fdust, 0.35 * out, out)
        inh = np.where(fdust, 0.30, _INH[kind])
        jit = _JIT[kind]
        vxy = (inh[:, None] * vc[None, :]
               + (side * out)[:, None] * nr
               + np.column_stack([(U[:, 5] - 0.5) * 2.0 * jit,
                                  (U[:, 6] - 0.5) * 2.0 * jit]))
        # sub-frame birth: a particle born u*dt ago left the wheel where the
        # wheel WAS then and has flown since -- an even trail at any fps
        age0 = U[:, 8] * dt
        pxy = base + (vxy - vc[None, :]) * age0[:, None]
        pz = z0 + up * age0

        self.pos[slots] = np.column_stack([pxy, pz])
        self.vel[slots] = np.column_stack([vxy, up])
        self.age[slots] = age0
        self.life[slots] = life
        self.r0[slots] = r0
        self.r1[slots] = r1
        self.inten[slots] = ie
        self.kind[slots] = kind
        self.var[slots] = (self._serial + np.arange(n)) % N_VAR
        self._serial += n
        self.par[slots] = self._pblock[kind]
        self.alive[slots] = True
        # a recycled slot is a NEW particle: no drawn history
        self._drawn[slots] = False
        self._drawn_pp[slots] = False
        self._cut_p[slots] = False
        self.emitted += np.bincount(kind, minlength=N_KINDS)

    def _derive(self) -> None:
        """Radius (ease-out growth) and alpha (fade in, fade out, dilution)."""
        par = self.par
        v = 1.0 - np.minimum(self.age / self.life, 1.0)     # 1 - u, life left
        self.rad = self.r0 + (self.r1 - self.r0) * (1.0 - v * v)
        fin = np.minimum(self.age / par[:, P_FIN], 1.0)
        a = par[:, P_APK] * self.inten * fin * v ** par[:, P_FOUT] \
            * (self.r0 / np.maximum(self.rad, 1e-6)) ** par[:, P_DIL]
        a[~self.alive] = 0.0
        self.alp = a

    # ------------------------------------------------------------------ #
    #  THE JOLT                                                           #
    # ------------------------------------------------------------------ #
    def _update_jolt(self, rnd, V, surfs, dt, paused) -> None:
        n_kerb = sum(1 for s in surfs if s == 'kerb')
        n_grass = sum(1 for s in surfs if s == 'grass')
        n_grav = sum(1 for s in surfs if s == 'gravel')
        tk = 0.0 if n_kerb == 0 else (0.6 if n_kerb == 1 else 1.0)
        tr = min(1.0, (n_grass + 1.25 * n_grav) / 4.0)
        if dt > 0.0:
            for attr, tgt in (('_env_k', tk), ('_env_r', tr)):
                e = getattr(self, attr)
                tau = JOLT_ATTACK if tgt > e else JOLT_RELEASE
                e += (tgt - e) * (1.0 - math.exp(-dt / tau))
                if tgt == 0.0 and e < JOLT_SNAP:
                    e = 0.0
                setattr(self, attr, e)
            sv = _clip01((V - V_JOLT_LO) / (V_JOLT_HI - V_JOLT_LO))
            self._ph_k += (JOLT_KERB_HZ[0] + (JOLT_KERB_HZ[1] - JOLT_KERB_HZ[0]) * sv) * dt
            gfrac = n_grav / max(n_grass + n_grav, 1)
            f_lo = JOLT_GRASS_HZ[0] + (JOLT_GRAVEL_HZ[0] - JOLT_GRASS_HZ[0]) * gfrac
            f_hi = JOLT_GRASS_HZ[1] + (JOLT_GRAVEL_HZ[1] - JOLT_GRASS_HZ[1]) * gfrac
            self._ph_r += (f_lo + (f_hi - f_lo) * sv) * dt
        if (paused or getattr(rnd, '_cam3', None) is None
                or (self._env_k == 0.0 and self._env_r == 0.0)):
            self._jolt = (0.0, 0.0)     # plan views: a map does not shake
            return
        gate = _clip01((V - V_WALK) / V_WALK)          # nothing at a walk
        sv = _clip01((V - V_JOLT_LO) / (V_JOLT_HI - V_JOLT_LO))
        ak = (JOLT_KERB_M[0] + (JOLT_KERB_M[1] - JOLT_KERB_M[0]) * sv) * self._env_k * gate
        ar = (JOLT_ROUGH_M[0] + (JOLT_ROUGH_M[1] - JOLT_ROUGH_M[0]) * sv) * self._env_r * gate
        sd = self.seed
        pk = self._ph_k
        # the kerb: a regular ridge rhythm (the sine) roughened by noise;
        # |.| <= 1 each, weights sum to 1, so |dz_kerb| <= ak
        dz = ak * (0.6 * math.sin(2.0 * math.pi * pk)
                   + 0.4 * _vnoise(1.37 * pk, sd + 1))
        dz += ar * _vnoise(self._ph_r, sd + 2)
        dx = ar * JOLT_LAT * _vnoise(0.8 * self._ph_r + 17.0, sd + 3)
        self._jolt = (dx, dz)

    def jolt(self) -> tuple:
        """(dx right, dz up) metres of camera shake for this frame. The
        renderer applies it relative to the pose (Chase3D.jolt), so it never
        accumulates; (0, 0) in the plan views and on tarmac."""
        return self._jolt

    # ------------------------------------------------------------------ #
    #  DRAWING                                                            #
    # ------------------------------------------------------------------ #
    def draw(self, rnd, x: float, y: float, psi: float, far: bool) -> None:
        """far=True: chase -- particles deeper than the car; plan -- all.
        far=False: chase only, the particles between the eye and the car."""
        t0 = time.perf_counter()
        if not self.enabled:
            return
        if far or self._prep_frame != self._frame:
            self._prepare(rnd, x, y, psi)
        items = self._far_items if far else self._near_items
        if items:
            rnd.screen.blits(items, doreturn=False)
        lines = self._far_lines if far else self._near_lines
        if lines:
            self._draw_lines(rnd.screen, lines)
        self.ms_draw += (time.perf_counter() - t0) * 1e3

    @staticmethod
    def _draw_lines(surf, lines) -> None:
        """Streaks, alpha-blended (gfxdraw blends an RGBA colour). A wide
        one is a second line one pixel across its direction."""
        gl = _gfx.line
        for x0, y0, x1, y1, col, wide in lines:
            gl(surf, x0, y0, x1, y1, col)
            if wide:
                if abs(x1 - x0) >= abs(y1 - y0):
                    gl(surf, x0, y0 + 1, x1, y1 + 1, col)
                else:
                    gl(surf, x0 + 1, y0, x1 + 1, y1, col)

    def _split_depth(self, rnd, x, y, psi, c3) -> float:
        """Camera depth of the car footprint's NEAREST corner.

        The contract's split is the car's CG depth; a puff at the rear wheel
        is ~1.5 m nearer than that and still BEHIND the bumper, so splitting
        at the CG would paint a newborn puff over the tailgate. Splitting at
        the nearest corner keeps it under the body until it has actually
        drifted past it (0.55 m, ~3 frames at 11 m/s relative)."""
        R = _render_mod()
        f0, f1 = float(c3._f[0]), float(c3._f[1])
        c, s = math.cos(psi), math.sin(psi)
        along = c * f0 + s * f1
        across = -s * f0 + c * f1
        d_off = min(R.CAR_X_REAR * along, R.CAR_X_FRONT * along) \
            - R.CAR_HALF_W * abs(across)
        return float(getattr(rnd, '_car_depth', 0.0)) + d_off

    def _haze_levels(self, dep) -> np.ndarray:
        """(n,) haze level 0..N_HAZE-1 per camera depth (world.haze_factor)."""
        hq = np.zeros(dep.size, dtype=np.int64)
        hn = getattr(_world, 'HAZE_NEAR', 0.0)
        far_i = np.flatnonzero(dep > hn)
        if far_i.size:
            hmax = max(getattr(_world, 'HAZE_MAX', 1.0), 1e-6)
            for j in far_i.tolist():
                h = _world.haze_factor(float(dep[j]))
                hq[j] = int(round(h / hmax * (N_HAZE - 1)))
        return hq

    def _prepare(self, rnd, x, y, psi) -> None:
        """Project, cull, size, fade, sort and key this frame's particles,
        once; the far and near passes blit the two halves."""
        self._prep_frame = self._frame
        self._far_items = []
        self._near_items = []
        self._far_lines = []
        self._near_lines = []
        self.stat_cov = 0.0
        self.stat_px = 0
        self.stat_lines = 0
        self.stat_make_us = 0.0
        self.stat_made = 0
        W, H = int(rnd.W), int(rnd.H)
        if (W, H) != self._wh:
            self._setup_res(W, H)
        shown = self.alive & (self.alp > 0.5 / N_ALEV)
        drawn_now = np.zeros(self.n, dtype=bool)
        cut_now = np.zeros(self.n, dtype=bool)
        if not shown.any():
            self._roll_drawn(drawn_now, cut_now)
            return
        c3 = getattr(rnd, '_cam3', None)
        split = self._split_depth(rnd, x, y, psi, c3) if c3 is not None else 0.0
        streak = _STREAK[self.kind]
        js = np.flatnonzero(shown & streak)
        if js.size:
            self._prepare_lines(rnd, c3, js, split)
        idx = np.flatnonzero(shown & ~streak)
        if idx.size:
            self._prepare_puffs(rnd, c3, idx, split, W, H, drawn_now, cut_now)
        self._roll_drawn(drawn_now, cut_now)

    def _roll_drawn(self, drawn_now, cut_now) -> None:
        """Count blinks (drawn two frames ago, budget-cut last frame, drawn
        now) and shift the history."""
        self.blinks += int((self._drawn_pp & self._cut_p & drawn_now).sum())
        self._drawn_pp = self._drawn
        self._drawn = drawn_now
        self._cut_p = cut_now

    def _prepare_lines(self, rnd, c3, js, split) -> None:
        """Debris and drops: a streak from where the particle was STREAK_T
        ago, relative to the camera (which rides with the car), to where it
        is now. Opaque-ish lines, so no sprite, no budget, no cache."""
        P1 = self.pos[js]
        P0 = P1 - (self.vel[js] - self._vc) * STREAK_T
        a = self.alp[js]
        kd = self.kind[js]
        if c3 is None:
            S1 = rnd.world_to_screen(P1[:, :2])
            S0 = rnd.world_to_screen(P0[:, :2])
            near = np.zeros(js.size, dtype=bool)
            wide = np.full(js.size, self._wide)
            col = _LIT[kd]
        else:
            Q1 = c3.camera(P1)
            Q0 = c3.camera(P0)
            m = (Q1[:, 2] > STREAK_CULL) & (Q0[:, 2] > STREAK_CULL)
            if not m.any():
                return
            Q1, Q0, a, kd = Q1[m], Q0[m], a[m], kd[m]
            dep = Q1[:, 2]
            S1 = c3.project_cam(Q1)
            S0 = c3.project_cam(Q0)
            a = a * np.clip((dep - STREAK_CULL) / STREAK_FADE, 0.0, 1.0)
            near = dep < split
            wide = self._wide | (dep < STREAK_WIDE_DEPTH)
            col = _LIT[kd]
            hq = self._haze_levels(dep)
            if hq.any():
                h = (hq / (N_HAZE - 1) * getattr(_world, 'HAZE_MAX', 0.88))[:, None]
                haze = np.asarray(getattr(_world, 'HAZE_RGB', (188, 202, 216)),
                                  dtype=np.float64)
                col = col + (haze[None, :] - col) * h
        # no streak longer than STREAK_MAX_PX: shorten its tail
        dS = S0 - S1
        ln = np.hypot(dS[:, 0], dS[:, 1])
        lmax = STREAK_MAX_PX * self.res_k
        long_ = ln > lmax
        if long_.any():
            S0 = S0.copy()
            S0[long_] = S1[long_] + dS[long_] * (lmax / ln[long_])[:, None]
        W, H = rnd.W, rnd.H
        vis = (((S0[:, 0] >= 0) | (S1[:, 0] >= 0)) & ((S0[:, 0] < W) | (S1[:, 0] < W))
               & ((S0[:, 1] >= 0) | (S1[:, 1] >= 0)) & ((S0[:, 1] < H) | (S1[:, 1] < H)))
        al = np.clip(np.rint(a * 255.0), 0, 255).astype(np.int64)
        vis &= al >= 8
        if not vis.any():
            return
        ci = np.rint(col).astype(np.int64)
        s0 = np.rint(S0).astype(np.int64)
        s1 = np.rint(S1).astype(np.int64)
        far_l, near_l = self._far_lines, self._near_lines
        for j in np.flatnonzero(vis).tolist():
            ln = (int(s0[j, 0]), int(s0[j, 1]), int(s1[j, 0]), int(s1[j, 1]),
                  (int(ci[j, 0]), int(ci[j, 1]), int(ci[j, 2]), int(al[j])),
                  bool(wide[j]))
            (near_l if near[j] else far_l).append(ln)
        self.stat_lines = len(far_l) + len(near_l)

    def _prepare_puffs(self, rnd, c3, idx, split, W, H, drawn_now, cut_now) -> None:
        """The soft puffs: sprites, capped in size and in total pixels."""
        cap = self.r_cap
        if c3 is None:
            S = rnd.world_to_screen(self.pos[idx, :2])
            rpx = self.rad[idx] * float(rnd.ppm)
            a = self.alp[idx]
            hq = np.zeros(idx.size, dtype=np.int64)
            # top-down: the higher particle is the nearer one
            order = np.argsort(self.pos[idx, 2], kind='stable')
            near = np.zeros(idx.size, dtype=bool)
            shaded = 0
        else:
            Q = c3.camera(self.pos[idx])
            dep = Q[:, 2]
            m = dep > NEAR_CULL
            if not m.any():
                return
            idx, Q, dep = idx[m], Q[m], dep[m]
            S = c3.project_cam(Q)
            rpx = self.rad[idx] * c3.fl / dep
            a = self.alp[idx] * np.clip((dep - NEAR_CULL) / NEAR_FADE, 0.0, 1.0)
            big = rpx > cap
            if big.any():
                a[big] *= (cap / rpx[big]) ** CAP_FADE
            hq = self._haze_levels(dep)
            order = np.argsort(-dep, kind='stable')
            near = dep < split
            shaded = 1
        rpx = np.clip(rpx, R_MIN_PX, cap)
        vis = ((S[:, 0] + rpx > 0.0) & (S[:, 0] - rpx < W)
               & (S[:, 1] + rpx > 0.0) & (S[:, 1] - rpx < H))
        b = np.clip(np.rint(np.log(rpx / R_MIN_PX) * self._inv_log_ratio),
                    0, N_BUCKETS - 1).astype(np.int64)
        alev = np.clip(np.rint(a * N_ALEV), 0, N_ALEV).astype(np.int64)
        vis &= alev > 0
        # the pixel budget: a four-wheel slide in the chase view puts ~40
        # puffs at the cap, 1.5 M px. Over budget, the faintest are dropped
        # (they are also the ones whose loss shows least) -- ranked with
        # hysteresis, so a puff at the cut-off does not change sides every
        # frame (a one-frame blink) while its neighbours' alpha jitters.
        area = self._dim[b] ** 2
        if float(area[vis].sum()) > self.px_budget:
            cand = np.flatnonzero(vis)
            ic = idx[cand]
            pri = a[cand] * np.where(self._drawn[ic], 1.0 + BUDGET_HYST,
                                     np.where(self._cut_p[ic], 1.0 / (1.0 + BUDGET_HYST), 1.0))
            rank = cand[np.argsort(-pri, kind='stable')]
            keep = rank[np.cumsum(area[rank]) <= self.px_budget]
            vis[:] = False
            vis[keep] = True
            cut_now[idx[cand]] = True
            cut_now[idx[keep]] = False
            self.budget_drops += int(cand.size - keep.size)
        drawn_now[idx[vis]] = True
        order = order[vis[order]]
        if order.size == 0:
            return
        self.stat_cov = float(np.sum(a[order] * rpx[order] ** 2)) * math.pi / (W * H)
        self.stat_px = int(area[order].sum())
        kd = self.kind[idx]
        keys = (((((kd * N_VAR + self.var[idx]) * N_BUCKETS + b)
                  * (N_ALEV + 1) + alev) * N_HAZE + hq) * 2 + shaded)
        ctr = self._ctr[b]
        tlx = np.rint(S[:, 0] - ctr).astype(np.int64)
        tly = np.rint(S[:, 1] - ctr).astype(np.int64)
        dims = self._dim[b]
        cache = self._cache
        get, touch = cache.get, cache.move_to_end
        far_items, near_items = self._far_items, self._near_items
        for k, px, py, nr, d in zip(keys[order].tolist(), tlx[order].tolist(),
                                    tly[order].tolist(), near[order].tolist(),
                                    dims[order].tolist()):
            sp = get(k)
            if sp is None:
                sp = self._sprite(k)
                if sp is None:
                    self.make_skips += 1
                    continue
                w = sp.get_width()
                if w != d:                  # borrowed a neighbouring size
                    px -= (w - d) // 2
                    py -= (w - d) // 2
            else:
                touch(k)
            (near_items if nr else far_items).append((sp, (px, py)))

    # ------------------------------------------------------------------ #
    #  SPRITES                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode(key: int) -> tuple:
        """key -> (kind, var, b, alev, hq, shaded)."""
        k = key
        shaded = k % 2
        k //= 2
        hq = k % N_HAZE
        k //= N_HAZE
        alev = k % (N_ALEV + 1)
        k //= (N_ALEV + 1)
        b = k % N_BUCKETS
        k //= N_BUCKETS
        return k // N_VAR, k % N_VAR, b, alev, hq, shaded

    @staticmethod
    def _with(key: int, alev: int = None, db: int = 0) -> int:
        """The key with another alpha level and/or size bucket."""
        kind, var, b, al, hq, shaded = Effects._decode(key)
        if alev is not None:
            al = alev
        return (((((kind * N_VAR + var) * N_BUCKETS + (b + db))
                  * (N_ALEV + 1) + al) * N_HAZE + hq) * 2 + shaded)

    def _profile(self, var: int, b: int) -> np.ndarray:
        """(d, d) float32 alpha profile 0..1, surfarray [x, y] layout.

        A puff is a soft core plus four lumps set round it at a per-variant
        angle, max-combined: a cauliflower outline instead of a perfect disc,
        which is most of what makes a blob read as smoke."""
        key = (var, b)
        p = self._prof.get(key)
        if p is not None:
            return p
        R = float(self._rb[b])
        d = int(self._dim[b])
        c = (d - 1) * 0.5
        xx = (np.arange(d, dtype=np.float32) - c)[:, None]
        yy = (np.arange(d, dtype=np.float32) - c)[None, :]
        rr = np.sqrt(xx * xx + yy * yy)
        u = rr / R
        p = np.clip(1.0 - u * u, 0.0, 1.0) ** 2.2
        if R >= 4.0:
            th0 = 0.35 + 0.9 * var
            for j, w in enumerate((1.0, 0.8, 0.9, 0.7)):
                th = th0 + j * 0.5 * math.pi + 0.25 * math.sin(3.1 * j + var)
                ox, oy = 0.40 * R * math.cos(th), 0.40 * R * math.sin(th)
                q = ((xx - ox) ** 2 + (yy - oy) ** 2) / (0.55 * R) ** 2
                lump = np.clip(1.0 - q, 0.0, 1.0) ** 2.0 * (0.85 * w)
                np.maximum(p, lump, out=p)
        p = p.astype(np.float32)
        self._prof[key] = p
        return p

    def _colour(self, kind: int, hq: int, shaded: int, b: int) -> np.ndarray:
        """(3,) plan colour, or (1, d, 3) chase ramp: lit on top, the sky-blue
        underside below (the sun is high: world.SUN_DIR z 0.70), tinted
        toward world.HAZE_RGB at the haze level."""
        lit, shade = _LIT[kind], _SHADE[kind]
        h = hq / (N_HAZE - 1) * getattr(_world, 'HAZE_MAX', 0.88)
        haze = np.asarray(getattr(_world, 'HAZE_RGB', (188, 202, 216)), dtype=np.float64)
        if not shaded:
            col = 0.8 * lit + 0.2 * shade
            return col + (haze - col) * h
        R = float(self._rb[b])
        d = int(self._dim[b])
        c = (d - 1) * 0.5
        w = np.clip(0.35 + 0.65 * (np.arange(d) - c) / max(R, 1.0), 0.0, 1.0)
        ramp = lit[None, :] * (1.0 - w[:, None]) + shade[None, :] * w[:, None]
        ramp = ramp + (haze[None, :] - ramp) * h
        return ramp[None, :, :]

    def _put(self, key: int, sp) -> None:
        """Insert into the LRU cache and evict down to its bounds."""
        cache = self._cache
        cache[key] = sp
        self._cache_px += sp.get_width() * sp.get_height()
        while self._cache_px > self.cache_px_max or len(cache) > CACHE_MAX:
            _, old = cache.popitem(last=False)
            self._cache_px -= old.get_width() * old.get_height()

    def _borrow(self, key: int):
        """A cached stand-in for a sprite the frame may not build: the
        nearest alpha level of the same sprite, else a size up to two
        buckets (~40 %) away at a nearby level; None if nothing close is
        cached, and the puff sits out a frame. Measured over a harsh mixed
        session (5 surfaces x 3 zooms, teleported between): puffs sit out
        only in the first 1-20 frames of a new surface, never after."""
        _, _, b, alev, _, _ = self._decode(key)
        get = self._cache.get
        for dl in range(1, N_ALEV + 1):
            for al2 in (alev + dl, alev - dl):
                if 0 < al2 <= N_ALEV:
                    sp = get(self._with(key, al2))
                    if sp is not None:
                        return sp
        for db in (-1, 1, -2, 2):
            if 0 <= b + db < N_BUCKETS:
                for dl in (0, 1, -1, 2, -2, 3, -3):
                    al2 = alev + dl
                    if 0 < al2 <= N_ALEV:
                        sp = get(self._with(key, al2, db))
                        if sp is not None:
                            return sp
        return None

    def _sprite(self, key: int):
        """Build (or borrow) the sprite for a key. The full-alpha 'base' is
        built from scratch; every other level is derived from it (module
        docstring: 125 against 19 us at 162 px). Past MAKE_MAX / MAKE_US_MAX
        this frame -- one build is always allowed -- borrow instead."""
        kind, var, b, alev, hq, shaded = self._decode(key)
        d = int(self._dim[b])
        bkey = self._with(key, N_ALEV)
        base = self._cache.get(bkey)
        cost = 0.0
        if base is None:
            cost += MAKE_US_BASE[0] + MAKE_US_BASE[1] * d * d
        if alev != N_ALEV:
            cost += MAKE_US_DERIVED[0] + MAKE_US_DERIVED[1] * d * d
        if self.stat_made > 0 and (self.stat_made >= MAKE_MAX
                                   or self.stat_make_us + cost > MAKE_US_MAX):
            return self._borrow(key)
        self.stat_made += 1
        self.stat_make_us += cost
        prof = self._profile(var, b)
        if base is None:
            self.sprites_made += 1
            col = self._colour(kind, hq, shaded, b)
            base = pygame.Surface((d, d), pygame.SRCALPHA)
            px = pygame.surfarray.pixels3d(base)
            px[...] = np.clip(col, 0.0, 255.0).astype(np.uint8)
            del px
            pa = pygame.surfarray.pixels_alpha(base)
            np.multiply(prof, 255.0, out=pa, casting='unsafe')
            del pa
            self._put(bkey, base)
        else:
            self._cache.move_to_end(bkey)
        if alev == N_ALEV:
            return base
        self.sprites_made += 1
        sp = base.copy()
        pa = pygame.surfarray.pixels_alpha(sp)
        np.multiply(prof, 255.0 * alev / N_ALEV, out=pa, casting='unsafe')
        del pa
        self._put(key, sp)
        return sp


# ======================================================================= #
#  SELF-CHECK                                                             #
# ======================================================================= #
def _rig_smoke() -> dict:
    """Drive drive/vehicle.py's own rigs for every car in cars.CAR_ORDER
    and feed them through Effects.update on tarmac, with the car's key: the
    smoke onset against each car's grip limit, a lock-up, wheelspin, and
    ABS stops replayed at several frame rates. Imports vehicle and cars
    lazily -- the renderer never does."""
    from types import SimpleNamespace
    from . import vehicle as VH
    import cars as CARS

    class _Rnd:
        def __init__(self):
            self.cfg = None
            self.track = None
            self._t_render = 0.0
            self._st = None
            self._cam3 = None

    DTP = VH.DT_PHYS
    EVERY = 16                      # physics steps a frame (62.5 fps)
    MU_WET = (0.632,) * 4           # track.MU_WET_SCALE, the lowest on-road mu

    def row(veh):
        return (float(veh.x), float(veh.y), float(veh.psi), float(veh.u),
                float(veh.v), np.array(veh.alpha, dtype=float),
                np.array(veh.kappa, dtype=float), float(veh.util_f),
                float(veh.util_r))

    def feed(e, r, rw, key, frame_dt):
        r._t_render += frame_dt
        r._st = SimpleNamespace(u=rw[3], v=rw[4])
        aux = SimpleNamespace(V=rw[3], alpha=rw[5], kappa=rw[6], util_f=rw[7],
                              util_r=rw[8], surf4=('tarmac',) * 4,
                              on_track=True, paused=False, time_scale=1.0,
                              car_key=key)
        e.update(r, rw[0], rw[1], rw[2], aux, None)

    def ramp(key, wing, V, past):
        """vehicle.ramp_steer's rig, rate, peak and abort (|beta| > 12 deg,
        or a_y 3 % down after 1 s), fed every 16 ms and run on `past` s
        after the peak -> (puffs up to the peak a_y, puffs up to the front
        axle's own grip peak, puffs at the end, peak g, s run past it)."""
        cfg = VH.VehicleConfig(wing=wing)
        veh = VH._rig(CARS.get(key), cfg, V, +1)
        ctl = VH.Controls(wing_on=wing != 'off')
        r = _Rnd()
        e = Effects(r)
        rate = math.radians(2.2)
        best = uf_best = -1.0
        n = n_pk = n_uf = 0
        t_pk = 0.0
        done = False
        k = 0
        while True:
            ctl.delta = rate * k * DTP
            veh.step(ctl, VH._MU1, VH._MU1, DTP)
            k += 1
            t = k * DTP
            if k % EVERY == 0:
                feed(e, r, row(veh), key, EVERY * DTP)
                n = int(e.emitted[K_SMOKE])
            if not done:
                ay = abs(veh.ay)
                if ay > best:
                    best, n_pk, t_pk = ay, n, t
                if veh.util_f > uf_best:
                    uf_best, n_uf = veh.util_f, n
                done = (abs(veh.beta) > math.radians(12.0)
                        or (ay < 0.97 * best and t > 1.0))
            if (done and t - t_pk >= past) or t > 30.0:
                break
        return n_pk, n_uf, n, best / 9.81, t - t_pk

    def record(key, cfg, ctl, mu, V0, gear, steps, stop_v=None, dump=None):
        """1 ms trace of a straight-line run (a stop down to `stop_v`, or
        `steps` steps with the clutch dumped at step `dump`)."""
        veh = VH.Vehicle(CARS.get(key), cfg)
        veh.reset(V=V0, gear=gear)
        out = []
        for k in range(1, steps + 1):
            if dump is not None and k == dump:
                ctl.clutch = 0.0
            veh.step(ctl, mu, VH._MU1, DTP)
            if dump is None or k >= dump:
                out.append(row(veh))
            if stop_v is not None and veh.u <= stop_v:
                break
        return out

    def replay(tr, key, fps, jitter=0.0, seed=0):
        """The trace fed at `fps` with +-jitter frame lengths (seeded) ->
        smoke puffs."""
        rng = np.random.default_rng(seed)
        r = _Rnd()
        e = Effects(r)
        k = 0
        while True:
            jit = rng.uniform(-jitter, jitter) if jitter else 0.0
            s = max(1, int(round(1000.0 / fps * (1.0 + jit))))
            k += s
            if k > len(tr):
                break
            feed(e, r, tr[k - 1], key, s * DTP)
        return int(e.emitted[K_SMOKE])

    # the speed at which each car's |alpha| up to the peak is largest
    # (wing off): the tightest case for its onset
    V_WORST = {'corsa': 40.0, 'mx5': 40.0, '540i': 12.0}
    ABS_FPS = ((144, 0.0, 0), (60, 0.0, 0), (30, 0.0, 0),
               (24, 0.5, 1), (24, 0.5, 2), (24, 0.5, 3))
    out = {'ramp': {}, 'plate22': {}, 'abs': {}, 'lock': {}, 'spin': {}}
    for key in CARS.CAR_ORDER:
        # 1. ramp steer: nothing up to the peak a_y; the plough after it
        out['ramp'][key] = (V_WORST[key],) + ramp(key, 'off', V_WORST[key], 2.5)
        #    ... and with the plate at 22 m/s (a tyre-limited peak)
        out['plate22'][key] = ramp(key, 'plate', 22.0, 0.0)
        # 2. ABS stop from 30 m/s, full pedal, down to the smoke speed gate
        tr = record(key, VH.VehicleConfig(abs_on=True),
                    VH.Controls(brake=1.0, clutch=1.0, auto_gearbox=False),
                    VH._MU1, 30.0, 5, 20000, stop_v=V_SMOKE_MIN)
        kmax = max(float(np.abs(w[6]).max()) for w in tr)
        out['abs'][key] = ([replay(tr, key, f, j, sd) for f, j, sd in ABS_FPS],
                           kmax, len(tr) * DTP)
        # 3. a lock-up: ABS off, full pedal from 25 m/s, 1 s at 62.5 fps
        tr = record(key, VH.VehicleConfig(abs_on=False),
                    VH.Controls(brake=1.0, clutch=1.0, auto_gearbox=False),
                    VH._MU1, 25.0, 4, 1000)
        out['lock'][key] = replay(tr, key, 62.5)
        # 4. wheelspin: the clutch dumped in first at 3 m/s, full throttle
        #    (the 540i bogs down from rest), 2 s
        tr = record(key, VH.VehicleConfig(),
                    VH.Controls(throttle=1.0, auto_clutch=False, clutch=1.0),
                    VH._MU1, 3.0, 1, 2800, dump=800)
        out['spin'][key] = (replay(tr, key, 62.5),
                            max(float(np.abs(w[6]).max()) for w in tr))
    # 5. the worst on-road ABS case: the 540i at mu 0.632 from 15 m/s
    #    (above 0.40 for 84 ms at a time)
    tr = record('540i', VH.VehicleConfig(abs_on=True),
                VH.Controls(brake=1.0, clutch=1.0, auto_gearbox=False),
                MU_WET, 15.0, 4, 20000, stop_v=V_SMOKE_MIN)
    out['abs_wet'] = ([replay(tr, '540i', f, j, sd) for f, j, sd in ABS_FPS],
                      max(float(np.abs(w[6]).max()) for w in tr))
    # 6. a WING-CARRIED peak: past the fronts' own grip peak the panel keeps
    #    a_y rising, so smoke may come before the peak a_y -- never before
    #    the fronts' grip peak
    out['carried'] = {'corsa fin 32': ramp('corsa', 'fin', 32.0, 0.0),
                      '540i plate 40': ramp('540i', 'plate', 40.0, 0.0)}
    return out


def self_check(verbose: bool = True, screenshot_dir: str = 'runs') -> bool:
    """Emission by slip and surface, the smoke onset against the physics'
    grip limit, the pool cap, ageing, determinism, robustness, the jolt
    (zero / bounded / not cumulative), and the draw budget, coverage and
    stability in car_up and chase at 1280x800 and 1920x1080 through a
    headless Renderer, with screenshots."""
    import os
    from types import SimpleNamespace
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    from . import render as R
    from . import track as trk

    ok_all = True
    res = []

    def rep(tag, passed, msg=''):
        nonlocal ok_all
        ok_all = ok_all and bool(passed)
        res.append((tag, bool(passed), msg))
        if verbose:
            print(f'  [{"ok" if passed else "FAIL"}] {tag:46s} {msg}')

    if verbose:
        print('drive/fx.py self-check')
    tr = trk.make_arena()
    DT = 1.0 / 60.0

    def pose_at(s, n=0.0, dpsi=0.0):
        # the exact curve, not the 0.5 m polyline samples: a snapped pose
        # jerks the chase camera back and forth every frame
        x, y, p, _ = trk._eval(tr, s)
        return (float(x) - n * math.sin(p), float(y) + n * math.cos(p), float(p) + dpsi)

    def hud(V, slide=0.0, surf=None, on_track=True):
        """slide 1 = a real slide: the rears at 16 deg (the fronts plough to
        15 deg 2 s past the grip peak), kappa 0.30 held."""
        a = R._demo_hud(V=V)
        a.wing_side, a.wing_deploy = 0, 0.0
        a.surf4 = surf
        a.on_track = on_track
        if slide > 0.0:
            a.kappa = np.array([0.02, 0.02, 0.30 * slide, 0.30 * slide])
            a.alpha = np.radians([-5.0, -5.0, -16.0 * slide, -16.0 * slide])
            a.util_r = 1.0
        return a

    def slide4(V=16.0):
        a = hud(V, 1.0, ('tarmac',) * 4)
        a.alpha = np.radians([-18.0, -18.0, -18.0, -18.0])
        a.util_f = 1.0
        return a

    class _FakeRnd:
        """What Effects.update reads, without a screen: a plan view."""
        def __init__(self):
            self.cfg = R.ViewConfig()
            self.track = tr
            self._t_render = 0.0
            self._st = None
            self._cam3 = None

    def run(fx, fr, aux, frames, s0=300.0, n0=0.0, dpsi=0.0, slide_v=0.0):
        V = _f(aux.V, 16.0)
        for k in range(frames):
            s = s0 + V * DT * k
            x, y, p = pose_at(s, n0, dpsi)
            fr._t_render += DT
            fr._st = R._demo_state(x, y, p, u=V, v=-slide_v, r=0.0)
            fx.update(fr, x, y, p, aux, None)

    # ---- 1. emission by slip and by surface ----------------------------
    fr = _FakeRnd()
    fx = Effects(fr)
    run(fx, fr, hud(25.0), 120)
    rep('no emission while gripping (tarmac, 25 m/s)', fx.n_alive == 0
        and int(fx.emitted.sum()) == 0,
        f'{int(fx.emitted.sum())} emitted, demo HUD alpha 6.5 deg, kappa 0.05')
    fx.reset()
    run(fx, fr, hud(16.0, 1.0, ('tarmac',) * 4), 45, slide_v=2.0)
    c = fx.counts()
    rep('smoke from a slide on tarmac', c['smoke'] >= 20 and
        sum(c.values()) == c['smoke'], str({k: v for k, v in c.items() if v}))
    fx.reset()
    run(fx, fr, hud(16.0, 1.0, ('kerb', 'tarmac', 'kerb', 'tarmac')), 45)
    c = fx.counts()
    rep('... and on a kerb', c['smoke'] > 0, str({k: v for k, v in c.items() if v}))
    fx.reset()
    run(fx, fr, hud(18.0, 0.0, ('grass',) * 4, on_track=False), 60, n0=9.0)
    c = fx.counts()
    rep('grass: dust and grass bits, no smoke', c['dust'] > 0 and c['grass'] > 0
        and c['smoke'] == 0 and c['gravel'] == 0,
        str({k: v for k, v in c.items() if v}))
    fx.reset()
    run(fx, fr, hud(18.0, 0.0, ('gravel',) * 4, on_track=False), 60, n0=9.0)
    c = fx.counts()
    rep('gravel: beige dust and stones', c['gravel'] > 0 and c['stone'] > 0
        and c['dust'] == 0 and c['smoke'] == 0,
        str({k: v for k, v in c.items() if v}))
    fx.reset()
    run(fx, fr, hud(24.0, 0.0, ('wet',) * 4), 60)
    c = fx.counts()
    n_fast = c['spray'] + c['drop']
    rep('wet: mist and drops, nothing else', c['spray'] > 0 and c['drop'] > 0
        and sum(c.values()) == n_fast, str({k: v for k, v in c.items() if v}))
    fx.reset()
    run(fx, fr, hud(10.0, 0.0, ('wet',) * 4), 60)
    c = fx.counts()
    rep('... more spray with speed', c['spray'] + c['drop'] < n_fast,
        f'{c["spray"] + c["drop"]} at 10 m/s against {n_fast} at 24 m/s')
    fx.reset()
    a_fb = hud(18.0, 0.0, None, on_track=False)
    run(fx, fr, a_fb, 40, n0=9.0)
    c1 = fx.counts()
    fx.reset()
    a_fb = hud(24.0, 0.0, None)
    a_fb.mu = np.array([0.632, 0.632, 0.632, 0.632])
    run(fx, fr, a_fb, 40)
    c2 = fx.counts()
    rep('surf4 None: off track -> grass, mu 0.63 -> wet',
        c1['dust'] > 0 and c1['smoke'] == 0 and c2['spray'] > 0 and c2['smoke'] == 0,
        f'off: {c1["dust"]} dust; wet: {c2["spray"]} spray')
    fx.reset()
    run(fx, fr, hud(1.2, 1.0, ('grass',) * 4, on_track=False), 60, n0=9.0)
    rep('nothing at walking pace', int(fx.emitted.sum()) == 0,
        '1.2 m/s, sliding, on grass')
    rg, ig = fx._rates(18.0, ('grass',) * 4, np.zeros(4))
    rs_, is_ = fx._rates(18.0, ('grass',) * 4, np.ones(4))
    rep('grass in a straight line: the rears throw the dust',
        abs(rg[0, K_DUST] / rg[2, K_DUST] - 0.25) < 1e-9
        and ig[0, K_DUST] < ig[2, K_DUST] and rs_[0, K_DUST] == rs_[2, K_DUST],
        f'front dust rate x{rg[0, K_DUST] / rg[2, K_DUST]:.2f} of the rears '
        f'gripping, x{rs_[0, K_DUST] / rs_[2, K_DUST]:.2f} sliding')

    # ---- 2. the smoke onset against the physics' grip limit ----------------
    rig = _rig_smoke()
    rp, p22 = rig['ramp'], rig['plate22']
    rep('no smoke up to the peak ay, every car',
        all(v[1] == 0 for v in rp.values()) and all(v[0] == 0 for v in p22.values()),
        'ramp steer, wing off: ' + ', '.join(
            f'{k} {v[0]:.0f} m/s {v[4]:.3f} g {v[1]}' for k, v in rp.items())
        + '; plate 22 m/s: ' + ' / '.join(str(v[0]) for v in p22.values())
        + ' puffs up to it')
    rep('... and smoke once the fronts plough past it',
        all(v[3] - v[1] >= 5 for v in rp.values()),
        ' / '.join(f'{k} {v[3] - v[1]}' for k, v in rp.items())
        + ' puffs in the 2.5 s after the peak')
    cr = rig['carried']
    rep('a wing-carried peak: none before the fronts\' grip peak',
        all(v[1] == 0 for v in cr.values()),
        ', '.join(f'{k}: {v[1]} (by its peak a_y {v[0]})' for k, v in cr.items()))
    ab, aw = rig['abs'], rig['abs_wet']
    rep('no smoke in an ABS stop (144 60 30 24j fps)',
        all(not any(v[0]) for v in ab.values()) and not any(aw[0]),
        ' / '.join(f'{k} {sum(v[0])}' for k, v in ab.items())
        + ' puffs, 30->2 m/s dry, kappa peaks '
        + '/'.join(f'{v[1]:.2f}' for v in ab.values())
        + f'; 540i mu 0.63 from 15 m/s {sum(aw[0])} (kappa {aw[1]:.2f})')
    rep('smoke from a lock-up (ABS off), every car',
        all(v >= 10 for v in rig['lock'].values()),
        ' / '.join(f'{k} {v}' for k, v in rig['lock'].items())
        + ' puffs in 1 s from 25 m/s')
    rep('smoke from wheelspin (a clutch dump), every car',
        all(v[0] >= 3 for v in rig['spin'].values()),
        ' / '.join(f'{k} {v[0]} (kappa {v[1]:.2f})' for k, v in rig['spin'].items())
        + ' puffs')

    # ---- 3. the cap, ageing, pausing, determinism, robustness ---------------
    fx.reset()
    a4 = slide4()
    peak, over = 0, False
    for _ in range(10):
        run(fx, fr, a4, 60)
        peak = max(peak, fx.n_alive)
        over |= fx.n_alive > fx.n
    rep('the pool cap holds in a 10 s four-wheel slide', not over and peak == fx.n,
        f'peak {peak} of {fx.n}, {int(fx.emitted.sum())} emitted')
    arr0 = [a.copy() for a in fx.state()]
    ap = slide4()
    ap.paused = True
    run(fx, fr, ap, 30)
    rep('nothing ages or emits while paused',
        all(np.array_equal(a, b) for a, b in zip(arr0, fx.state())), '30 frames')
    n_before = fx.n_alive
    run(fx, fr, hud(16.0), int(3.0 / DT) + 2)
    rep('particles age and die', n_before > 0 and fx.n_alive == 0,
        f'{n_before} -> {fx.n_alive} after 3 s of grip (longest life 2.6 s)')

    def replay():
        f = _FakeRnd()
        e = Effects(f)
        run(e, f, hud(16.0, 1.0, ('tarmac', 'tarmac', 'grass', 'gravel'),
                      on_track=False), 90, slide_v=2.0)
        run(e, f, hud(24.0, 0.5, ('wet',) * 4), 60)
        return e, f
    e1, f1 = replay()
    e2, f2 = replay()
    same = all(np.array_equal(a, b) for a, b in zip(e1.state(), e2.state()))
    e1.reset()
    f1._t_render = 0.0
    run(e1, f1, hud(16.0, 1.0, ('tarmac', 'tarmac', 'grass', 'gravel'),
                    on_track=False), 90, slide_v=2.0)
    run(e1, f1, hud(24.0, 0.5, ('wet',) * 4), 60)
    same_reset = all(np.array_equal(a, b) for a, b in zip(e1.state(), e2.state()))
    rep('deterministic: same frames -> identical pool', same and same_reset
        and e2.n_alive > 0, f'{e2.n_alive} live; reset() reproduces it too')

    fl = _FakeRnd()
    fl.cfg = R.ViewConfig(detail='low')
    el = Effects(fl)
    peak_l = 0
    for _ in range(4):
        run(el, fl, a4, 60)
        peak_l = max(peak_l, el.n_alive)
    fh = _FakeRnd()
    eh = Effects(fh)
    run(eh, fh, a4, 30)
    el.reset()
    fl._t_render = 0.0
    run(el, fl, a4, 30)
    rep("detail 'low': half the pool, half the emission",
        el.n == POOL_N_LOW and peak_l == POOL_N_LOW
        and int(el.emitted.sum()) <= int(np.ceil(0.5 * eh.emitted.sum())) + 4,
        f'pool {el.n}, peak {peak_l}; {int(el.emitted.sum())} against '
        f'{int(eh.emitted.sum())} emitted in 0.5 s')
    fh.cfg = R.ViewConfig(effects=False)
    run(eh, fh, a4, 30)
    rep('effects switched off at run time: empty, nothing emitted',
        eh.n_alive == 0 and not eh.enabled, 'ViewConfig.effects = False')

    bad = 0
    why = ''
    for tag, kw in (('V None', dict(V=None)), ('V nan', dict(V=float('nan'))),
                    ('util None', dict(util_f=None, util_r=None)),
                    ('time_scale None', dict(time_scale=None)),
                    ('alpha None x4', dict(alpha=[None] * 4)),
                    ('kappa nan', dict(kappa=np.array([np.nan] * 4))),
                    ('alpha scalar', dict(alpha=0.3)),
                    ('mu None x4', dict(mu=[None] * 4, surf4=None)),
                    ('wheel_xy junk', dict(wheel_xy=[[1, 2], [3]])),
                    ('surf4 short', dict(surf4=('grass',))),
                    ('beta None', dict(beta_deg=None)),
                    ('car_key junk', dict(car_key=None, car_name=3.5)),
                    ('car_key unknown', dict(car_key='F40', car_name=None))):
        try:
            f = _FakeRnd()
            e = Effects(f)
            a = slide4()
            for k_, v_ in kw.items():
                setattr(a, k_, v_)
            run(e, f, a, 5)
            f._st = SimpleNamespace(u=None, v=float('inf'))
            e.update(f, 1.0, 2.0, 0.3, a, None)
        except Exception as ex:            # noqa: BLE001 -- the point
            bad += 1
            why += f' {tag}: {type(ex).__name__}'
    rep('None / non-finite inputs never raise', bad == 0, why or '13 cases')

    # ---- 4. the jolt ---------------------------------------------------------
    rc = R.Renderer(R.ViewConfig(mode='chase'), tr, headless=True)

    def drive(r_, aux, frames, s0=330.0, n0=5.4):
        out = []
        V = float(aux.V)
        for k in range(frames):
            x, y, p = pose_at(s0 + V * DT * k, n0)
            st = R._demo_state(x, y, p, u=V, v=0.0, r=0.0)
            # the harness's shake strength (task 27: the Shake setting, the
            # pause, the wheels): on, so the renderer applies fx's jolt
            r_.update_camera(st, DT, shake=1.0)
            r_.draw_frame(st, None, 0.0, R._demo_ctl(), aux, None)
            out.append((r_._fx.jolt(), r_))
        return out

    js = drive(rc, hud(17.0, 0.0, ('tarmac',) * 4), 30)
    z_tar = max(max(abs(j[0]), abs(j[1])) for (j, _) in js)
    eye_ok = np.array_equal(rc._chase.eye, rc._chase._eye0)
    rep('jolt: exactly zero on tarmac', z_tar == 0.0 and eye_ok, f'max {z_tar:.3g} m')
    js = drive(rc, hud(17.0, 0.0, ('kerb', 'tarmac', 'kerb', 'tarmac')), 60)
    dz = np.array([j[1] for (j, _) in js[10:]])
    dx = np.array([j[0] for (j, _) in js[10:]])
    rep('jolt on a kerb: a vertical buzz, 1-2.5 cm',
        0.004 < np.abs(dz).max() <= JOLT_KERB_M[1] + 1e-12 and np.abs(dx).max() == 0.0,
        f'|dz| max {np.abs(dz).max() * 100:.2f} cm, rms {np.sqrt((dz ** 2).mean()) * 100:.2f} cm, '
        f'{int(np.sum(np.diff(np.sign(dz)) != 0))} sign changes / s')
    jx, jz = rc._fx.jolt()
    exp_eye = rc._chase._eye0 + jx * rc._chase._r + jz * rc._chase._u
    eye1 = rc._chase.eye.copy()
    rc.draw_frame(R._demo_state(*pose_at(330.0 + 17.0 * DT * 59, 5.4), u=17.0, v=0.0, r=0.0),
                  None, 0.0, R._demo_ctl(), hud(17.0, 0.0, ('kerb',) * 4), None)
    rep('jolt not cumulative (eye = pose + offset; a redraw adds 0)',
        np.allclose(eye1, exp_eye, atol=1e-12, rtol=0.0)
        and np.allclose(rc._chase.eye, eye1, atol=1e-12, rtol=0.0),
        f'offset {np.linalg.norm(eye1 - rc._chase._eye0) * 100:.2f} cm from the pose')
    js = drive(rc, hud(18.0, 0.0, ('grass',) * 4, on_track=False), 90, n0=9.0)
    dz = np.array([j[1] for (j, _) in js[15:]])
    dx = np.array([j[0] for (j, _) in js[15:]])
    rep('jolt on grass: 3-6 cm bumps, a little lateral',
        0.01 < np.abs(dz).max() <= JOLT_ROUGH_M[1] + 1e-12
        and 0.0 < np.abs(dx).max() <= JOLT_LAT * JOLT_ROUGH_M[1] + 1e-12,
        f'|dz| max {np.abs(dz).max() * 100:.2f} cm, |dx| max {np.abs(dx).max() * 100:.2f} cm')
    js = drive(rc, hud(18.0, 0.0, ('tarmac',) * 4), 90, n0=0.0)
    rep('jolt releases to exactly zero back on tarmac',
        js[-1][0] == (0.0, 0.0) and np.array_equal(rc._chase.eye, rc._chase._eye0),
        f'after {90 * DT:.1f} s (release tau {JOLT_RELEASE} s)')
    rp = R.Renderer(R.ViewConfig(mode='car_up'), tr, headless=True)
    js = drive(rp, hud(17.0, 0.0, ('kerb',) * 4), 30)
    #  the anchor moves by task 27's shake_offset alone -- fx adds nothing
    sx_, sy_ = rp._shake_px
    want_ = rp._anchor0 + np.array([sx_, sy_]) * rp.ui
    rep('plan views: no fx jolt (the anchor moves by task 27\'s shake alone)',
        all(j == (0.0, 0.0) for (j, _) in js) and np.allclose(rp._anchor, want_, atol=1e-9),
        f'fx jolt 0; anchor off the pose by task 27\'s ({sx_:+.2f}, {sy_:+.2f}) px')

    # ---- 5. drawing: a full pool, timed, at two resolutions -----------------
    calib = R.cpu_calibration()
    shots = {}
    cov = {}
    for mode, size, budget, p99b in (('car_up', (1280, 800), 0.6, 1.5),
                                     ('chase', (1280, 800), 1.0, 2.5),
                                     ('chase', (1920, 1080), 1.2, 3.0)):
        rr = R.Renderer(R.ViewConfig(mode=mode, size=size), tr, headless=True)
        e = rr._fx
        a_s = slide4()
        ts, full, cv, px, made = [], [], [], [], []
        for k in range(330):
            x, y, p = pose_at(300.0 + 16.0 * DT * k, -1.0, 0.25)
            st = R._demo_state(x, y, p, u=16.0 * math.cos(0.25),
                               v=-16.0 * math.sin(0.25), r=0.0)
            rr.update_camera(st, DT if k else 0.0)
            rr.draw_frame(st, None, 0.0, R._demo_ctl(), a_s, None)
            made.append((e.stat_made, e.stat_make_us))
            if k >= 150:
                ts.append(e.ms_update + e.ms_draw)
                full.append(e.n_alive)
                cv.append(e.stat_cov)
                px.append(e.stat_px)
        tag = f'{mode}_{size[0]}'
        path = os.path.join(os.path.abspath(screenshot_dir), f'fx_smoke_{tag}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rr.screenshot(path)
        shots[tag] = path
        cov[tag] = float(np.mean(cv))
        t = np.array(ts)
        okb, why = R.frame_budget_verdict(float(t.mean()), float(np.percentile(t, 99)),
                                          budget_mean=budget, budget_p99=p99b,
                                          calib=calib)
        rep(f'{mode} {size[0]}x{size[1]}: effects cost, full pool',
            okb and min(full) >= 0.85 * e.n,
            f'{min(full)}-{max(full)} live, {np.mean(px) / 1e6:.2f} M px; {why}')
        mk = np.array(made)
        rep('... sprite builds capped per frame',
            int(mk[:, 0].max()) <= MAKE_MAX
            and float(np.max(mk[:, 1][mk[:, 0] > 1], initial=0.0)) <= MAKE_US_MAX,
            f'{e.sprites_made} built, max {int(mk[:, 0].max())} / frame '
            f'(est {mk[:, 1].max():.0f} us), {e.make_skips} puff-frames skipped; '
            f'cache {len(e._cache)} ({e._cache_px / 1e6:.2f} M px)')
        if mode == 'chase':
            rep('... no budget blinks (hysteresis)', e.blinks == 0,
                f'{e.blinks} drawn-cut-drawn in {len(ts)} frames, '
                f'{e.budget_drops / len(ts):.1f} drops / frame')
    ratio = cov['chase_1920'] / max(cov['chase_1280'], 1e-9)
    rep('1920x1080 looks like 1280x800 (smoke coverage)', 0.8 <= ratio <= 1.25,
        f'alpha-weighted coverage {cov["chase_1280"]:.3f} at 800 rows, '
        f'{cov["chase_1920"]:.3f} at 1080 (x{ratio:.2f}; fixed-px caps gave x0.34)')

    # the car stays visible in plan: the particles are under it
    rq = R.Renderer(R.ViewConfig(mode='car_up'), tr, headless=True)
    rq0 = R.Renderer(R.ViewConfig(mode='car_up', effects=False), tr, headless=True)
    a_d = hud(18.0, 1.0, ('grass',) * 4, on_track=False)
    path = os.path.join(os.path.abspath(screenshot_dir), 'fx_dust_car_up.png')
    for k in range(90):
        x, y, p = pose_at(640.0 + 18.0 * DT * k, 9.0, 0.35)
        st = R._demo_state(x, y, p, u=18.0, v=-1.2, r=0.2)
        # the headless renderers share ONE display surface: read each frame
        # back before the other renderer draws over it
        for r_ in (rq, rq0):
            r_.update_camera(st, DT)
            r_.draw_frame(st, None, 0.0, R._demo_ctl(), a_d, None)
            if k == 89:
                if r_ is rq:
                    A = pygame.surfarray.array3d(r_.screen)
                    rq.screenshot(path)
                else:
                    B = pygame.surfarray.array3d(r_.screen)
    cx_, cy_ = (int(v) for v in rq.world_to_screen(np.array([x, y])))
    win = (slice(cx_ - 3, cx_ + 4), slice(cy_ - 3, cy_ + 4))
    n_diff = int(np.any(A != B, axis=2).sum())
    rep('plan: dust drawn, the car on top of it',
        n_diff > 500 and np.array_equal(A[win], B[win]),
        f'{n_diff} px changed, the car centre untouched ({rq._fx.n_alive} live)')
    shots['dust'] = path

    # the streaks: grass bits and stones are lines, never sprites
    rs = R.Renderer(R.ViewConfig(mode='chase', shake=False), tr, headless=True)
    a_g = hud(18.0, 1.0, ('grass', 'grass', 'gravel', 'gravel'), on_track=False)
    nl = 0
    for k in range(90):
        x, y, p = pose_at(640.0 + 18.0 * DT * k, 9.0, 0.2)
        st = R._demo_state(x, y, p, u=18.0 * math.cos(0.2), v=-18.0 * math.sin(0.2), r=0.0)
        rs.update_camera(st, DT if k else 0.0)
        rs.draw_frame(st, None, 0.0, R._demo_ctl(), a_g, None)
        nl = max(nl, rs._fx.stat_lines)
    streak_keys = sum(1 for kk in rs._fx._cache
                      if not _PUFFY[Effects._decode(kk)[0]])
    path = os.path.join(os.path.abspath(screenshot_dir), 'fx_offroad_chase.png')
    rs.screenshot(path)
    shots['offroad'] = path
    rep('bits and stones drawn as streaks, not sprites', nl > 0 and streak_keys == 0,
        f'up to {nl} streaks a frame; {len(rs._fx._cache)} sprites, none a bit')

    # banned exact colours: the HUD signal colours never appear in the world
    banned = [(217, 206, 85), (78, 194, 106), (226, 82, 63), (176, 78, 224),
              (120, 220, 160)]
    hits = 0
    n_sp = 0
    for sp in (list(rq._fx._cache.values()) + list(rr._fx._cache.values())
               + list(rs._fx._cache.values())):
        n_sp += 1
        px = pygame.surfarray.pixels3d(sp)
        for c_ in banned:
            hits += int(np.all(px == np.array(c_, dtype=np.uint8), axis=2).sum())
        del px
    haze = np.asarray(getattr(_world, 'HAZE_RGB', (188, 202, 216)), dtype=np.float64)
    for kd in np.flatnonzero(_STREAK):
        for h in np.linspace(0.0, 1.0, 256):
            cl = tuple(int(v) for v in np.rint(_LIT[kd] + (haze - _LIT[kd]) * h))
            hits += int(cl in banned)
    rep('no sprite or streak pixel is a HUD signal colour', hits == 0,
        f'{n_sp} sprites, {int(_STREAK.sum())} streak ramps')
    rep('screenshots', all(os.path.exists(p) for p in shots.values()),
        ', '.join(shots.values()))
    if verbose:
        print(f'  {"PASS" if ok_all else "FAIL"}: '
              f'{sum(1 for _, p, _ in res if p)}/{len(res)} checks')
    return ok_all


if __name__ == '__main__':
    sys.exit(0 if self_check() else 1)
