"""drive/props.py -- the solid things round the road.

Trees, barriers, fences, stands, buildings, boards, gantries: laid out from
the track's own geometry (so a generated track gets them too) and drawn in
both the plan views and the chase view by `render.Renderer.draw_frame`'s
hooks. Read-only with respect to physics; does not import `render` at module
level (render imports it); it is handed the renderer.

Why this module looks the way it does
-------------------------------------
THE CAR HAS NO COLLISION WITH SCENERY, and the physics must not change to give
it one. So a prop the car can reach reads as a bug -- the car drives through a
grandstand -- and the whole layout is built round one rule: every solid prop
stands >= CLEAR_SOLID = 30 m from the nearest track edge (distance from the
centreline minus width / 2) and >= CLEAR_AREA = 10 m outside every
`Track.areas` polygon. 30 m is further than a spin carries the car on the
arena's measured exits, and it is past the ground agent's run-off, which ends
at RUNOFF_MAX = 24 m. Five things cannot be 30 m away and still be what they
are; each is its own clearance class, measured by the self-check:

  * the start / finish GANTRY's pillars, 4 m: a gantry spans the road, and a
    pillar 30 m out would make it a 72 m bridge. Its beam is OVERHEAD (6.3 m,
    against a 1.44 m car), so only the pillars are ground footprints.
  * BRAKE and TIMING boards, 6 m: a board 30 m out is unreadable at 30 m/s,
    which is the only reason a board exists.
  * the dragstrip's concrete WALLS, 8 m: a drag strip is walled, and a wall
    30 m out is a field boundary, not a strip wall.
  * the dragstrip's START-LIGHT tree, 3 m: the driver reads it off the line.

Everything is placed by the same `_Ctx.place` test (footprint sampled every
<= 1 m against the exact point-to-segment distance of a 1 m centreline, the
areas' own polygons, and every footprint already placed), so a track the
layout was never tuned for gets fewer props rather than props on the road.
The self-check measures it on all seven maps and on a generated circuit the
themes have never seen: min solid prop 30.5 m (arena), board 6.8, gantry
pillar 4.8, dragstrip wall 8.8, start-light tree 3.8, tree canopy 36.2.

THE LAYOUT IS GENERIC. It reads the centreline, the curvature (corners are
runs of |kappa| > 1/200 m^-1; a SLOW corner has R <= 65 m and gets brake boards
and a stand; a kink of R > 100 m does not end a braking zone; a HALF-TURN of
150 deg or more is both whatever its radius -- the oval's R150 bends, task
46), the straights,
the start line (s = 0: the gantry, the pits and the main stand go on the
straight that contains it), the sector lines and `Track.gates`, with a
per-map THEME picked by `track.name` ('arena' circuit, 'open' proving ground,
'skidpad', 'dragstrip'; anything else is a circuit). Randomness is
`np.random.default_rng(zlib.crc32(name))`: the same track is the same layout,
bit for bit, and the self-check hashes it.

THE FRAME BUDGET IS THE CONSTRAINT (60 fps, the physics in the same frame).
Measured on this machine, a small `draw.polygon` is 0.8 us, a 100x80 px one
4.7 us and a colour-keyed RLE blit of a 150 px sprite 1.2 us (per-pixel alpha:
12.3 us), while `ndarray.tolist()` of 3000x3 floats is 0.32 ms. So the chase
view is ONE pre-built world-space polygon soup (render.Mesh's layout: one
vertex block, starts / counts / normals, per-polygon colour with the SUN
shading precomputed at build) and a frame is a fixed number of numpy calls:

  1. cull OBJECTS (chunks of <= 20 m) on centre +- radius against the depth
     range and the frustum -- one (M, 3) matmul;
  2. per polygon: its object's verdict, a level-of-detail window on the
     object depth, and the backface test;
  3. gather the survivors' vertices and project them in ONE call; the few
     polygons that reach nearer than NEAR_Z (beside or behind the eye) or
     past the clamp band are clipped to the camera's own frustum (_clip_cam,
     render.Chase3D's planes, ~5 us each; 0-2 a frame on the road, <= 8 off
     it) -- dropping them lost up to 740 000 px2 of a frame 40 m off the
     arena (a pit wall, the shadow the camera sat in), and clamping them
     bent their edges;
  4. haze each polygon by its object's depth: a TALL object (> LAND_Z) on
     the land curve that ends in the world's painted tree line
     (_land_haze), a low one and every shadow on the ground band's own
     curve (world.haze_factor); over the last FADE_FRAC of its range a
     polygon fades OUT with alpha (gfxdraw: 0.3-0.6 us at that size), so it
     dissolves into whatever is behind it instead of popping or turning
     into a pale ghost of haze colour;
  5. one lexsort: ground shadows first, then objects far to near, decals
     after their face, polygons far to near; the tree sprites are merged
     into the same order;
  6. split at `rnd._car_depth`: far=True draws what is deeper than the car
     (and every ground shadow), far=False the rest, after the car -- by the
     object's centre, except that a face of an object straddling the car's
     depth goes after the car exactly when its plane separates the eye from
     the car; and a polygon painted after the car that reaches nearer than
     Z_OCC and overlaps the car on screen fades out, so a board the camera
     is about to pass through never hides the car.

Trees are pre-rendered SPRITES (5 species, 8 shapes, lit from the front, the
side or behind depending on each tree's bearing to the sun, so the lit side
sweeps across a wood as the camera turns instead of every tree flipping at
once), smoothscaled once per size bucket (x1.15 apart), colour-keyed with RLE
through a pygame Mask (9 us a sprite) and hazed per bucket of HAZE_STEPS with
two SDL fills (14 us); in the last FADE_FRAC of R_TREE a tree steps down
through FADE_STEPS of surface alpha on the same RLE surface (0.45 us a blit
against 0.38 opaque). A tree is one blit, addressed by one int computed for
every visible tree in one numpy expression. The far land is thinner and
mostly copses: the arena's start frame first drew 346 trees, 311 of them past
200 m, where a tree is 30-60 px -- now 265.

Level of detail, all on the OBJECT depth so an object never half-switches:
crowd speckle to LOD_CROWD (115 m; it was 375 of that frame's 726 polygons
out to 170 m), one quad per tyre-wall chunk past LOD_NEAR (150 m), posts and
lamp dots to LOD_POSTS (90 m), thin tops to LOD_TOP (70 m). cfg.detail 'low'
scales every range by DETAIL_LOW_R, drops half the trees and the alpha
shadows, and costs 0.6-0.7x in the chase view.

MEASURED, frames interleaved with the previous build of this module in one
process (1280x800, dummy driver, machine x1.07 of the calibration reference):
the busiest arena chase frame (the start: main stand + pits, ~480 polygons,
~270 trees) spends 1.93 ms mean in here (1.87 before the clipping, the two
haze curves and the fades), the other chase scenes 0.48-1.53 ms, the plan
views 0.02-0.73 ms, and 0.81 ms mean / 1.38 max at the widest manual zoom.
Driving 33-60 m off the road, among the props, the worst frame is 3.6 ms
(10-14 ms before the big-sprite rework). The layout is ~140 ms and the soup
~60 ms per track (cached per process by track geometry); the sprites ~110 ms
plus the haze / fade warm-up, once per process.

The plan views draw the same objects as FOOTPRINTS: the soup's up-facing
polygons (so a gabled or barrel roof's lit and shaded slopes come for free,
and a flat roof gets a parapet stroke, lit on its sun-facing edges), drop
shadows along -SUN_DIR, barrier / wall / fence strokes, boards as bars, trees
as shadow + canopy + highlight circles.

The shared look (sun, haze, grass) is drive/world.py's: every colour here is
lit by world.SUN_DIR and fades into world.HAZE_RGB through world.haze_factor
(bent to the painted land's airiness for tall props: _land_haze).
Nothing here impersonates a real brand: the sponsor boards are abstract
colour blocks, the flags are plain fields, the vehicles generic boxes.

`python3 -m drive.props` is the self-check.
"""
from __future__ import annotations

import math
import os
import sys
import time
import zlib

import numpy as np
import pygame
import pygame.gfxdraw

from . import world as _world

# ======================================================================= #
#  CONSTANTS                                                              #
# ======================================================================= #
# --- clearance: the car has no collision with scenery (see the docstring)
CLEAR_SOLID = 30.0       # m  every solid prop, from the nearest track edge
CLEAR_GANTRY = 4.0       # m  the start gantry's pillars (the beam is overhead)
CLEAR_BOARD = 6.0        # m  brake / timing boards: unreadable further out
CLEAR_WALL = 8.0         # m  the dragstrip's concrete walls
CLEAR_XMAS = 3.0         # m  the dragstrip's start-light tree
CLEAR_AREA = 10.0        # m  outside every Track.areas polygon (all classes)
CLEAR = {'solid': CLEAR_SOLID, 'gantry': CLEAR_GANTRY, 'board': CLEAR_BOARD,
         'wall': CLEAR_WALL, 'xmas': CLEAR_XMAS}
RUNOFF_MAX = 24.0        # m  the ground agent's gravel / run-off ends here: an
                         #    OPAQUE grass-coloured shadow is only right beyond
                         #    it; nearer, shadows are alpha-blended
PLACE_MARGIN = 0.6       # m  placement keeps this much over the class minimum,
                         #    so a 0.5 m centreline chord can never fail it
SAG_MAX = 0.3            # m  a barrier chunk's chord may stray this far from
                         #    the offset line it replaces (< PLACE_MARGIN)

# --- corners (curvature runs; see the docstring)
K_CORNER = 1.0 / 200.0   # 1/m  |kappa| above this is a corner
R_SLOW = 65.0            # m    a corner this tight gets brake boards / a stand
R_KINK = 100.0           # m    a corner gentler than this does not end a
                         #      braking zone (the arena's T7, R130)
#: ... but a corner that turns this far is braked for, walled with tyres and
#: watched from a stand whatever its radius (task 46): the oval's two R150
#: half-turns, 480 m straights in, would otherwise read as kinks -- no
#: boards, no tyre wall, no corner stand. Every other circuit's corners of
#: 150 deg or more (Linden's T2, Ashdown's T4, the hairpin test's) are
#: R <= 65 already, so their layouts are bit for bit unchanged
TURN_HALF_DEG = 150.0

# --- the chase view
NEAR_Z = 1.0             # m  a polygon with every vertex deeper than this is
                         #    projected directly; one reaching nearer (beside
                         #    or behind the eye) is CLIPPED to the camera's
                         #    own frustum (_clip_cam)
LOD_NEAR = 150.0         # m  full detail inside this object depth
LOD_SPECKLE = 170.0      # m  decals (sponsor shapes, bands) out to here
LOD_CROWD = 115.0        # m  crowd speckle out to here; one quad per row past it
                         #    (measured: the main stand was 375 of the start
                         #    frame's 726 polygons with speckle to 170 m)
LOD_TOP = 70.0           # m  thin tops (tyre walls, boards): < 1 px past this
LOD_POSTS = 90.0         # m  fence / armco posts out to here
R_BIG = 560.0            # m  draw range: stands, buildings, hangars, masts
R_MED = 380.0            # m  ... barriers, boards, walls, fences
R_SMALL = 240.0          # m  ... flags, cones, trailers, marshal posts
R_TREE = 430.0           # m  ... trees (the horizon must not be empty)
FADE_FRAC = 0.18         # -  over the last 18 % of a range a prop fades OUT
                         #    (alpha: into whatever is behind it -- the ground
                         #    haze, the painted tree line, the sky), so
                         #    nothing pops at the edge of its range
FADE_STEPS = 8           # -  alpha steps of a tree sprite in its fade band
LAND_HAZE = 0.42         # -  the haze a tall prop tends to: the panorama's
                         #    land (see _land_haze)
LAND_Z = 3.0             # m  an object taller than this stands against the
                         #    panorama and takes the LAND haze; a lower one
                         #    lies on the ground band and takes the ground's
Z_OCC = 1.5              # m  a polygon painted after the car that reaches
                         #    nearer than this and overlaps the car on screen
                         #    fades out (alpha) ...
Z_GONE = 0.6             # m  ... to nothing here (_prep_chase, 3c). Measured
                         #    over 10,504 poses 6.5-60 m off the road, four
                         #    headings: frames where such a polygon, opaque,
                         #    covered the car's screen centre went 62 -> 6
                         #    (the 6 are walls whose visible part is >= 1.7 m
                         #    away with the car inside the building behind
                         #    them). A faded one filling the frame is a 2 ms
                         #    gfxdraw fill: the worst props frame there went
                         #    5.4 -> 5.8 ms, the mean +0.03; nothing on the
                         #    road comes within 1.5 m of the eye
MAX_CLIP = 64            # -  near-plane / clamp-band clips per frame at most
                         #    (measured: <= 4 a frame on the road, <= 8 at
                         #    33-60 m off it, sideways and backwards)
FRUSTUM_PAD = 1.12       # -  frustum half-angles widened by this for culling
CLAMP_PX = 4096          # px a polygon reaching further past the frame is
                         #    clipped (render clamps at the same band)
DETAIL_LOW_R = 0.55      # -  cfg.detail 'low': detail ranges scaled by this

# --- the sun (world.SUN_DIR), precomputed onto every face at build
SUN = np.array(_world.SUN_DIR, dtype=np.float64)
SUN_XY = SUN[:2] / SUN[2]          # ground shadow offset per metre of height
HAZE = np.array(_world.HAZE_RGB, dtype=np.float64)
FOOT = np.array(_world.LAND_FOOT_RGB, dtype=np.float64)   # the ground at HAZE_LAND
AMB = 0.50               # est  shade of a face turned away from the sun
DIF = 0.58               # est  ... plus this times max(0, n . sun)
SKY = 0.06               # est  ... plus this times n_z: the sky's fill
POLE_SHADE = 0.80        # est  a vertical cylinder's mean shade
_GR = np.array(_world.GRASS_RGB, dtype=np.float64)
_GR2 = np.array(_world.GRASS2_RGB, dtype=np.float64)
SHADOW_RGB = tuple(int(v) for v in (0.5 * (_GR + _GR2) * 0.60))   # grass in shade
SHADOW_ALPHA = 92        # est  alpha of a shadow cast onto road / run-off

#: Colours the self-checks COUNT and that the world must therefore never draw:
#: C_YELLOW, C_GREEN, C_BAR_BRK, C_PURPLE, the PB ghost's green, and the three
#: tarmac tones render's `_road_px` counts as road. `_guard` nudges any computed
#: colour that lands on one by one blue level.
FORBIDDEN = np.array([(217, 206, 85), (78, 194, 106), (226, 82, 63),
                      (176, 78, 224), (120, 220, 160), (58, 61, 67),
                      (43, 58, 74), (51, 57, 63)], dtype=np.int32)

# --- palette (est: a late-afternoon summer day at a well-kept circuit)
C_CONCRETE = (196, 192, 182)
C_CONCRETE_DK = (150, 147, 140)
C_WHITE = (228, 228, 222)
C_ROOF = (170, 174, 178)
C_ROOF_DK = (112, 116, 122)
C_ROOF_RED = (150, 74, 58)
C_UNDER = (82, 84, 90)
C_GLASS = (62, 80, 102)
C_DOOR = (92, 96, 104)
C_STEEL = (176, 180, 184)
C_STEEL_DK = (108, 112, 118)
C_POLE = (150, 152, 156)
C_TYRE = (34, 34, 37)
C_TYRE_TOP = (48, 48, 52)
C_BELTS = ((184, 52, 44), (224, 224, 218), (44, 82, 150), (224, 224, 218))
C_SEAT = (66, 92, 136)
C_TREAD = (142, 140, 134)
C_BOARD_BACK = (120, 124, 130)
C_HANGAR = (184, 188, 190)
C_HANGAR_DOOR = (104, 110, 116)
C_OFFICE = (214, 206, 188)
C_CONE = (230, 120, 36)
C_LAMP_OFF = (70, 34, 32)
C_LAMP_RED = (200, 40, 36)
C_AMBER = (232, 150, 40)
C_GO = (70, 176, 84)
C_HOUSING = (30, 32, 36)
C_TRUNK = (88, 74, 60)

#: crowd shirt colours (est: muted, a summer crowd, plus empty seats)
CROWD = ((172, 58, 50), (58, 78, 132), (214, 208, 190), (46, 46, 52),
         (196, 146, 64), (104, 122, 74), (150, 150, 156), (122, 72, 104),
         (232, 228, 220), C_SEAT, C_SEAT)

#: FICTIONAL sponsor boards: (background, [(shape, colour)]) -- abstract
#: colour blocks only; no letters, no logos, nothing that reads as a brand
DESIGNS = (
    ((230, 230, 224), (('bar', (180, 44, 40)), ('disc', (36, 64, 140)))),
    ((30, 44, 78), (('chev', (232, 160, 44)),)),
    ((172, 42, 38), (('split', (236, 236, 230)),)),
    ((34, 88, 62), (('band', (230, 230, 224)), ('sq', (230, 230, 224)))),
    ((236, 236, 230), (('dots', None),)),
    ((214, 122, 42), (('zig', (36, 36, 44)),)),
    ((44, 46, 54), (('stripes', (206, 206, 200)),)),
    ((70, 120, 170), (('bar', (236, 236, 230)), ('sq', (214, 122, 42)))),
)
#: flag fields: plain two-tone fields, not any nation's flag
FLAGS = ((190, 50, 44), (40, 70, 140), (230, 230, 224), (40, 110, 80),
         (220, 150, 40), (60, 60, 70))

# --- trees ---------------------------------------------------------------
#: species: image aspect (w / h), height range m, plan canopy radius / height,
#: and the leaf palette (dark, mid, light). est: European summer trees.
SPECIES = {
    'broad':   dict(aspect=0.84, h=(9.0, 17.0), r=0.30,
                    cols=((38, 60, 30), (70, 100, 44), (124, 148, 64))),
    'conifer': dict(aspect=0.46, h=(11.0, 21.0), r=0.19,
                    cols=((24, 44, 32), (44, 74, 50), (88, 118, 74))),
    'poplar':  dict(aspect=0.30, h=(15.0, 23.0), r=0.13,
                    cols=((42, 64, 30), (78, 108, 46), (134, 156, 70))),
    'copse':   dict(aspect=1.55, h=(10.0, 15.0), r=0.62,
                    cols=((36, 56, 30), (66, 94, 42), (118, 142, 60))),
    'bush':    dict(aspect=1.70, h=(2.4, 4.2), r=0.62,
                    cols=((40, 60, 32), (72, 98, 46), (120, 142, 64))),
}
SPECIES_ORDER = ('broad', 'conifer', 'poplar', 'copse', 'bush')
SHAPES = {'broad': 2, 'conifer': 2, 'poplar': 1, 'copse': 2, 'bush': 1}
LIGHTS = ('F', 'L', 'R', 'K')      # front-lit, sun left, sun right, backlit
TREE_MASTER_H = 360      # px  master sprite height (upscaled past it: soft)
TREE_B0 = 3.0            # px  smallest size bucket
TREE_RATIO = 1.15        # -   bucket ratio: a 7 % size error is invisible
TREE_PREBUILD = 200      # px  buckets up to here are built with the sprites
TREE_PX_MAX = 900        # px  a tree nearer than this size is not drawn
TREE_IMG_K = 1.06        # -   image height / tree height (the base shadow)
TREE_FOOT = 0.035        # -   the trunk base sits this far up the image
TREE_NEAR = 4.0          # m   a tree base nearer than this is not drawn
TREE_SHADOW_R = 150.0    # m   chase: a tree's ground shadow is drawn out to here
TREE_SHADOW_N = 7        # -   ... as a 7-gon ellipse (the sprite has its own
                         #     contact shadow; this is the cast one)
HAZE_STEPS = 12          # -   haze buckets per sprite, over 0..LAND_HAZE
NSLOT = HAZE_STEPS * FADE_STEPS   # haze x fade slots per sprite (_TreeArt)
SLOT_PER_M = 2           # 1/m  the depth -> slot table's resolution
TREE_LOW_R = 0.8         # -   cfg.detail 'low': trees drawn to this x R_TREE
PLAN_THIN_PPM = 3.5      # px/m plan views zoomed out past this draw each tree
                         #      as one circle (no cast shadow, no highlight)
TREE_CACHE_BIG = 48      # -   lazily built big sprites kept (FIFO). 128 held
                         #     86-125 MB of pixels after a lap 45-60 m off the
                         #     road (a 924 px copse is 5.3 MB); 48 is ~40 MB
TREE_BIG_PER_FRAME = 1   # -   big sprites built per frame at most. Past the
                         #     master's height a bucket is a NEAREST scale of a
                         #     pre-keyed master: 1.1 ms + a 1.5 ms first (RLE)
                         #     blit for an 804 px copse, where smoothscale +
                         #     mask was 6.8 + 1.2 ms -- two of those a frame
                         #     were the 10-14 ms frames measured driving
                         #     through the woods. Past the budget a cached
                         #     neighbour bucket (+-15 % a step, up to 6 steps)
                         #     stands in for a frame or two
KEY = (255, 0, 255)      # the sprites' colour key


# ======================================================================= #
#  SMALL GEOMETRY                                                         #
# ======================================================================= #
def _guard(cols: np.ndarray) -> np.ndarray:
    """Nudge any colour that equals a FORBIDDEN one (int array, (K, 3))."""
    if len(cols):
        bad = (cols[:, None, :] == FORBIDDEN[None, :, :]).all(-1).any(-1)
        if bad.any():
            cols[bad, 2] = np.where(cols[bad, 2] < 255, cols[bad, 2] + 1, 254)
    return cols


def _render_mod():
    """drive.render, for its constants -- never imported at module level
    (render imports this module; by the time a frame is drawn it is
    loaded)."""
    m = sys.modules.get(__package__ + '.render') if __package__ else None
    if m is None:
        from . import render as m          # noqa: F811 -- lazy, by design
    return m


def _land_haze(h):
    """The haze of a TALL prop, given world.haze_factor at its depth.

    The ground band (world.draw_atmosphere) reaches HAZE_MAX = 0.88 at the
    horizon, but the land the world paints above it does not: its tree line
    runs from (84, 108, 88) at the crowns to (122, 142, 136) at the foot,
    which is this module's mean tree colour (52, 75, 35) hazed by 0.27 and
    0.54 (measured). A tree at 430 m on the ground curve was hazed 0.68 and
    faded on to the haze colour, so the far belts stood PALER than the
    painted line behind them, and the last trees were pale ghosts over it.
    Tall props therefore follow the ground curve while it is low and bend
    over to LAND_HAZE, the painted line's middle (tanh: slope 1 at 0; h 0.21
    -> 0.19 at 200 m, 0.41 -> 0.31 at 300 m, 0.68 -> 0.39 at 430 m), and
    the near belts, the far belts and the painted line read as one
    landscape. The last trees then fade OUT (alpha) over what is behind."""
    return LAND_HAZE * np.tanh(np.asarray(h, dtype=np.float64) / LAND_HAZE)


def _haze_tables():
    """(depths every 2 m, GROUND haze, LAND haze), tabulated from
    world.haze_factor so a change there is followed here without a second
    formula to keep in step, and both capped at world.HAZE_LAND (0.41, the
    haze at LAND_D = 300 m) as world.draw_atmosphere caps the ground band:
    past 300 m the ground under a prop is that tone, so a barrier at 360 m
    hazed 0.53 on the open curve stood as a pale line on a darker ground
    (the land curve tops out at 0.42 by itself; capped it ends where the
    ground does)."""
    d = np.arange(0.0, 1400.0, 2.0)
    h = np.array([_world.haze_factor(v) for v in d])
    cap = float(_world.HAZE_LAND)
    return d, np.minimum(h, cap), np.minimum(_land_haze(h), cap)


def _shade(rgb, n) -> tuple:
    """A face's colour under the sun, fixed at build (props never move)."""
    n = np.asarray(n, dtype=np.float64)
    g = AMB + DIF * max(0.0, float(n @ SUN)) + SKY * max(0.0, float(n[2]))
    return tuple(min(255, max(0, int(round(c * g)))) for c in rgb)


def _hull(P) -> np.ndarray:
    """Convex hull of (n, 2) points, counter-clockwise (monotone chain)."""
    pts = sorted(set((round(float(a), 4), round(float(b), 4)) for a, b in P))
    if len(pts) <= 2:
        return np.array(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1], dtype=np.float64)


def _shadow_of(P3) -> np.ndarray:
    """The ground shadow of a convex-ish solid: its points slid down the sun
    ray to z = 0, hulled with its footprint."""
    P3 = np.asarray(P3, dtype=np.float64)
    g = P3[:, :2] - P3[:, 2:3] * SUN_XY[None, :]
    return _hull(np.vstack([P3[:, :2], g]))


def _pip(P, poly) -> np.ndarray:
    """Vectorised even-odd point-in-polygon: (M, 2) points, (k, 2) polygon."""
    P = np.asarray(P, dtype=np.float64)
    x, y = P[:, 0][:, None], P[:, 1][:, None]
    a = np.asarray(poly, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    ya, yb = a[:, 1][None, :], b[:, 1][None, :]
    xa, xb = a[:, 0][None, :], b[:, 0][None, :]
    cond = (ya > y) != (yb > y)
    xc = xa + (y - ya) * (xb - xa) / np.where(np.abs(yb - ya) > 1e-12, yb - ya, 1e-12)
    return ((cond & (x < xc)).sum(axis=1) % 2) == 1


def _seg_dist(P, A, B, chunk: int = 512) -> np.ndarray:
    """min over segments A->B of the distance from each point P, exact."""
    P = np.asarray(P, dtype=np.float64)
    out = np.full(len(P), np.inf)
    if len(A) == 0 or len(P) == 0:
        return out
    D = B - A
    dd = np.maximum((D * D).sum(axis=1), 1e-12)
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk]
        px = p[:, 0][:, None] - A[:, 0][None, :]
        py = p[:, 1][:, None] - A[:, 1][None, :]
        t = np.clip((px * D[:, 0] + py * D[:, 1]) / dd, 0.0, 1.0)
        ex, ey = px - t * D[:, 0], py - t * D[:, 1]
        out[i:i + chunk] = np.sqrt((ex * ex + ey * ey).min(axis=1))
    return out


def _pairs(x, y) -> list:
    """(x, y) float arrays -> a list of int (x, y) TUPLES for pygame.

    Not `ndarray.tolist()` of an (N, 2) array: that makes N inner LISTS, and
    a list past the 80-entry free list is a fresh GC-tracked allocation.
    Measured over 300 arena chase frames, the props' ~2000 lists a frame took
    the collector from 24 gen-0 / 0 gen-2 passes to 297 / 2, and a gen-2 pass
    is a 4.3 ms stall of the whole process. Size-2 tuples come back off
    CPython's 2000-deep tuple free list, which does not count toward a
    collection."""
    return list(zip(np.asarray(x).astype(np.int32).tolist(),
                    np.asarray(y).astype(np.int32).tolist()))


def _quads(c, a) -> list:
    """(K, 3) int colours and (K,) alphas -> a list of RGBA tuples: one
    list for the opaque fills (draw.polygon ignores the alpha) and the
    alpha ones (gfxdraw), built without a per-polygon Python step."""
    return list(zip(c[:, 0].tolist(), c[:, 1].tolist(), c[:, 2].tolist(), a.tolist()))


def _clip_cam(P, zn, kx, ky) -> list:
    """Sutherland-Hodgman of a small camera-space polygon (a list of
    (x, y, z), x right, y up, z forward) against z >= zn and the four planes
    |x| <= z kx, |y| <= z ky through the eye; two points are a SEGMENT and
    are trimmed instead (Liang-Barsky). Pure Python: at 4-7 vertices it is
    ten times a numpy pass per plane."""
    planes = ((0.0, 0.0, 1.0, -zn), (-1.0, 0.0, kx, 0.0), (1.0, 0.0, kx, 0.0),
              (0.0, -1.0, ky, 0.0), (0.0, 1.0, ky, 0.0))
    if len(P) == 2:
        (ax, ay, az), (bx, by, bz) = P
        t0, t1 = 0.0, 1.0
        for a, b, c, d in planes:
            da = a * ax + b * ay + c * az + d
            db = a * bx + b * by + c * bz + d
            if da < 0.0 and db < 0.0:
                return []
            if da < 0.0:
                t0 = max(t0, da / (da - db))
            elif db < 0.0:
                t1 = min(t1, da / (da - db))
        if t1 <= t0:
            return []
        return [(ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az)) for t in (t0, t1)]
    for a, b, c, d in planes:
        ds = [a * x + b * y + c * z + d for x, y, z in P]
        if min(ds) >= 0.0:
            continue
        if max(ds) < 0.0:
            return []
        out = []
        n = len(P)
        for i in range(n):
            j = i + 1 if i + 1 < n else 0
            p, q, dp, dq = P[i], P[j], ds[i], ds[j]
            if dp >= 0.0:
                out.append(p)
            if (dp >= 0.0) != (dq >= 0.0):
                t = dp / (dp - dq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]),
                            p[2] + t * (q[2] - p[2])))
        P = out
    return P


def _cross2(a, b) -> float:
    """z of the 2-D cross product (numpy 2 deprecates np.cross on 2-vectors)."""
    return float(a[0] * b[1] - a[1] * b[0])


def _rect_pts(cx, cy, t, hl, hw) -> np.ndarray:
    """Corners of an oriented rectangle: centre, unit axis t, half extents."""
    tx, ty = t
    ox, oy = -ty, tx
    return np.array([(cx - hl * tx - hw * ox, cy - hl * ty - hw * oy),
                     (cx + hl * tx - hw * ox, cy + hl * ty - hw * oy),
                     (cx + hl * tx + hw * ox, cy + hl * ty + hw * oy),
                     (cx - hl * tx + hw * ox, cy - hl * ty + hw * oy)])


def _outline(poly, step: float = 1.0) -> np.ndarray:
    """A polygon's boundary sampled every <= `step` m, plus its centroid: what
    the clearance test measures (the nearest point of a footprint to a curved
    road can lie on an edge, not at a corner)."""
    poly = np.asarray(poly, dtype=np.float64)
    out = [poly.mean(axis=0)[None, :]]
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(math.ceil(math.hypot(*(b - a)) / step)))
        f = np.arange(n)[:, None] / n
        out.append(a[None, :] + f * (b - a)[None, :])
    return np.vstack(out)


def _sat(A, B) -> bool:
    """Do two convex polygons overlap (separating axis test)?"""
    for P in (A, B):
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            n = np.array([a[1] - b[1], b[0] - a[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


# ======================================================================= #
#  THE TRACK, READ AS GEOMETRY                                            #
# ======================================================================= #
class _Edge:
    """Distance from the nearest TRACK EDGE, m (negative on the ribbon).

    Exact: point-to-segment against the centreline resampled every ~1 m (a
    1 m chord on the tightest corner, R = 30, sags 4 mm), minus width / 2.
    Fast: point distance to a ~4 m resampling minus 2 m -- a LOWER bound
    (the nearest centreline point is within 2 m of a coarse sample), used to
    thin thousands of tree candidates before anything exact is paid for.
    The skidpad's guide circles count as road: drivers use them
    (`Track.guide_radii` about `Track.centre`). The areas are separate,
    `area()`: signed, negative inside.
    """

    def __init__(self, tr):
        self.tr = tr
        n = len(tr.xy) - 1 if tr.closed else len(tr.xy)
        xy = np.asarray(tr.xy[:n], dtype=np.float64)
        k = max(1, int(round(1.0 / tr.ds)))
        fine = xy[::k]
        if tr.closed:
            self.A, self.B = fine, np.roll(fine, -1, axis=0)
        else:
            if not np.allclose(fine[-1], xy[-1]):
                fine = np.vstack([fine, xy[-1:]])
            self.A, self.B = fine[:-1], fine[1:]
        kc = max(1, int(round(4.0 / tr.ds)))
        coarse = xy[::kc]
        if not tr.closed:
            coarse = np.vstack([coarse, xy[-1:]])
        self.C = coarse
        self.c_err = 0.5 * kc * tr.ds
        self.hw = 0.5 * float(tr.width)
        cen = getattr(tr, 'centre', ()) or ()
        self.rings = ([(float(cen[0]), float(cen[1]), float(R))
                       for R in (getattr(tr, 'guide_radii', []) or [])]
                      if len(cen) == 2 else [])
        self.areas = [np.asarray(a.polygon(), dtype=np.float64)
                      for a in (getattr(tr, 'areas', []) or [])]
        self._sbb = None

    def _rings(self, P) -> np.ndarray:
        d = np.full(len(P), np.inf)
        for cx, cy, R in self.rings:
            d = np.minimum(d, np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - R))
        return d

    def exact(self, P, cap: float | None = None) -> np.ndarray:
        """Exact edge distance. With `cap` (m past the EDGE), the answer is
        min(true, cap): the points are tiled 64 m square and each tile is
        measured only against the segments that can come within cap of it --
        a clearance test never needs more, and it is ~20x cheaper than every
        point against every segment."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        if cap is None or len(P) < 64:
            d = _seg_dist(P, self.A, self.B)
        else:
            c = float(cap) + self.hw
            d = np.full(len(P), c)
            if self._sbb is None:
                self._sbb = np.column_stack([np.minimum(self.A, self.B),
                                             np.maximum(self.A, self.B)])
            key = np.floor(P / 64.0).astype(np.int64)
            _u, inv = np.unique(key[:, 0] * 100003 + key[:, 1], return_inverse=True)
            inv = inv.ravel()
            for g in range(len(_u)):
                idx = np.flatnonzero(inv == g)
                q = P[idx]
                lo, hi = q.min(axis=0) - c, q.max(axis=0) + c
                m = ((self._sbb[:, 2] >= lo[0]) & (self._sbb[:, 0] <= hi[0])
                     & (self._sbb[:, 3] >= lo[1]) & (self._sbb[:, 1] <= hi[1]))
                if m.any():
                    d[idx] = np.minimum(_seg_dist(q, self.A[m], self.B[m]), c)
        d = np.minimum(d, self._rings(P))
        return d - self.hw

    def fast(self, P, chunk: int = 1024) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        out = np.empty(len(P))
        C = self.C
        for i in range(0, len(P), chunk):
            p = P[i:i + chunk]
            d2 = ((p[:, 0][:, None] - C[:, 0][None, :]) ** 2
                  + (p[:, 1][:, None] - C[:, 1][None, :]) ** 2)
            out[i:i + chunk] = np.sqrt(d2.min(axis=1))
        return np.minimum(out - self.c_err, self._rings(P)) - self.hw

    def area(self, P) -> np.ndarray:
        """Signed distance OUTSIDE the nearest area polygon (inf: no areas)."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        d = np.full(len(P), np.inf)
        for poly in self.areas:
            e = _seg_dist(P, poly, np.roll(poly, -1, axis=0))
            e = np.where(_pip(P, poly), -e, e)
            d = np.minimum(d, e)
        return d


def _nper(tr) -> int:
    return len(tr.xy) - 1 if tr.closed else len(tr.xy)


def _at(tr, s, n=0.0):
    """World (x, y), unit tangent (tx, ty) at arclength(s) s, offset n (LEFT
    positive): interpolated between the 0.5 m samples, wrapped on a closed
    track and clamped on an open one. Vectorised."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    n = np.broadcast_to(np.asarray(n, dtype=np.float64), s.shape)
    L = float(tr.length)
    ext = np.zeros_like(s)
    if tr.closed:
        s = np.mod(s, L)
    else:
        # an open track (the dragstrip) is EXTENDED along its end tangents,
        # so a wall can start behind the launch line
        ext = np.where(s < 0.0, s, np.where(s > L, s - L, 0.0))
        s = np.clip(s, 0.0, L)
    f = s / tr.ds
    N = len(tr.xy)
    i0 = np.clip(np.floor(f).astype(np.int64), 0, N - 2)
    w = (f - i0)[:, None]
    xy = tr.xy[i0] * (1.0 - w) + tr.xy[i0 + 1] * w
    c = np.cos(tr.psi[i0]) * (1.0 - w[:, 0]) + np.cos(tr.psi[i0 + 1]) * w[:, 0]
    sn = np.sin(tr.psi[i0]) * (1.0 - w[:, 0]) + np.sin(tr.psi[i0 + 1]) * w[:, 0]
    ln = np.maximum(np.hypot(c, sn), 1e-9)
    c, sn = c / ln, sn / ln
    P = np.column_stack([xy[:, 0] + ext * c - n * sn, xy[:, 1] + ext * sn + n * c])
    return P, np.column_stack([c, sn])


def _inside_sign(tr) -> float:
    """+1 if the infield is on the LEFT (a counter-clockwise loop)."""
    if not tr.closed:
        return 1.0
    return 1.0 if sum(sg.turn_deg for sg in tr.segs) >= 0.0 else -1.0


def _runs(mask, closed: bool) -> list:
    """(start, length) of the True runs of a boolean array, wrapping the seam
    of a closed track (a corner through s = 0 is ONE corner)."""
    m = np.asarray(mask, dtype=bool)
    n = len(m)
    if not m.any():
        return []
    if m.all():
        return [(0, n)]
    k0 = int(np.flatnonzero(~m)[0]) if closed else 0
    r = np.roll(m, -k0)
    d = np.diff(np.concatenate([[0], r.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int((a + k0) % n), int(b - a)) for a, b in zip(st, en)]


def _corners(tr) -> list:
    """Corners as dicts: s_in, s_out, R (min radius), sgn (+1 LEFT), s_mid,
    turn (deg, unsigned)."""
    n = _nper(tr)
    k = np.asarray(tr.kappa[:n], dtype=np.float64)
    out = []
    for a, ln in _runs(np.abs(k) > K_CORNER, tr.closed):
        idx = (a + np.arange(ln)) % n
        kk = k[idx]
        j = int(np.argmax(np.abs(kk)))
        s_in = float(tr.s[idx[0]])
        out.append(dict(s_in=s_in, s_out=s_in + ln * tr.ds,
                        R=1.0 / max(abs(float(kk[j])), 1e-9),
                        sgn=1.0 if kk[j] > 0 else -1.0,
                        s_mid=s_in + 0.5 * ln * tr.ds,
                        turn=abs(math.degrees(float(kk.sum()) * tr.ds))))
    out.sort(key=lambda c: c['s_in'])
    return out


def _straights(tr, min_len: float = 40.0) -> list:
    """(s0, s1) of the runs with |kappa| below a quarter of K_CORNER."""
    n = _nper(tr)
    k = np.abs(np.asarray(tr.kappa[:n], dtype=np.float64))
    out = []
    for a, ln in _runs(k < 0.25 * K_CORNER, tr.closed):
        if ln * tr.ds >= min_len:
            s0 = float(tr.s[a])
            out.append((s0, s0 + ln * tr.ds))
    return out


# ======================================================================= #
#  LAYOUT                                                                 #
# ======================================================================= #
class Prop:
    """One placed thing: a `kind`, a clearance `cls`, an oriented footprint
    (centre x y, unit axis t along its length, unit o = t rotated +90 deg,
    half extents hl hw) and a height. `o` is what the builders call OUTWARD
    for track-side props: the front faces -o. `extra` carries the rest."""

    __slots__ = ('kind', 'cls', 'x', 'y', 't', 'o', 'hl', 'hw', 'h', 'extra',
                 'foot')

    def __init__(self, kind, cls, x, y, t, hl, hw, h, **extra):
        self.kind, self.cls = kind, cls
        self.x, self.y = float(x), float(y)
        tx, ty = float(t[0]), float(t[1])
        ln = math.hypot(tx, ty) or 1.0
        self.t = (tx / ln, ty / ln)
        self.o = (-self.t[1], self.t[0])
        self.hl, self.hw, self.h = float(hl), float(hw), float(h)
        self.extra = extra
        self.foot = _rect_pts(self.x, self.y, self.t, self.hl, self.hw)

    def key(self) -> tuple:
        return (self.kind, self.cls, round(self.x, 3), round(self.y, 3),
                round(self.t[0], 4), round(self.t[1], 4), round(self.hl, 3),
                round(self.hw, 3), round(self.h, 3))


class _Ctx:
    """The layout under construction: the rng, the edge field, and every
    footprint placed so far (props may not overlap)."""

    def __init__(self, tr, detail: str = 'high'):
        self.tr = tr
        self.name = str(getattr(tr, 'name', '') or '')
        self.rng = np.random.default_rng(zlib.crc32(self.name.encode()))
        self.edge = _Edge(tr)
        self.props = []
        self.keep = []           # (poly, margin) tree keep-outs
        self.trees = []          # (P, H, Rc, species, shape) batches
        self._cxy = np.zeros((0, 2))
        self._crad = np.zeros(0)
        self.inside = _inside_sign(tr)
        self.hw = 0.5 * float(tr.width)

    # ------------------------------------------------------------------ #
    def clearance_ok(self, foot, cls: str) -> bool:
        P = _outline(foot)
        need = CLEAR[cls] + PLACE_MARGIN
        if float(self.edge.fast(P).min()) < need:
            if float(self.edge.exact(P, cap=need + 1.0).min()) < need:
                return False
        return float(self.edge.area(P).min()) >= CLEAR_AREA + PLACE_MARGIN

    def free(self, foot, gap: float = 1.0) -> bool:
        """No overlap with any placed footprint grown by `gap`: a vectorised
        bounding-disc prefilter, then the separating-axis test on the few
        that survive it."""
        n = len(self.props)
        if n == 0:
            return True
        if len(self._cxy) != n:
            self._cxy = np.array([(p.x, p.y) for p in self.props])
            self._crad = np.array([math.hypot(p.hl, p.hw) for p in self.props])
        c = foot.mean(axis=0)
        r = float(np.hypot(*(foot - c).T).max()) + gap
        d = np.hypot(self._cxy[:, 0] - c[0], self._cxy[:, 1] - c[1])
        for i in np.flatnonzero(d < r + self._crad + gap + 0.5):
            p = self.props[int(i)]
            grown = _rect_pts(p.x, p.y, p.t, p.hl + gap, p.hw + gap)
            if _sat(foot, grown):
                return False
        return True

    def place(self, prop: Prop, gap: float = 1.0, keep: float = 6.0) -> bool:
        """Add `prop` if it clears the road, the areas and every other prop."""
        if not self.clearance_ok(prop.foot, prop.cls):
            return False
        if not self.free(prop.foot, gap):
            return False
        self.props.append(prop)
        if keep > 0.0:
            self.keep.append((prop.foot, keep))
        return True

    def frame(self, s, side, off):
        """(x, y) of the point `off` m past the edge on `side` (+1 LEFT) at s,
        the along-track unit t, and o pointing AWAY from the track."""
        P, T = _at(self.tr, [s], side * (self.hw + off))
        t = T[0]
        o = side * np.array([-t[1], t[0]])
        return P[0], t, o

    def put(self, kind, cls, s, side, off, L, D, H, face_track=True,
            gap=1.0, keep=6.0, tries=(0.0, 4.0, 9.0, 16.0), **extra) -> Prop | None:
        """A rectangular prop whose FRONT (the -o face) is `off` m past the
        edge at s; pushed further out in `tries` steps until it fits."""
        for dt in tries:
            P, t, o = self.frame(s, side, off + dt + 0.5 * D)
            # the prop's own t is chosen so that its o (t rotated +90) is the
            # outward direction: then the -o face looks at the track
            tt = t if (side > 0) else -t
            if not face_track:
                tt = -tt
            p = Prop(kind, cls, P[0], P[1], tt, 0.5 * L, 0.5 * D, H, s=float(s),
                     side=float(side), **extra)
            if self.place(p, gap, keep):
                return p
        return None


# ----------------------------------------------------------------------- #
def _barrier_line(ctx, s0, s1, side, off, kind, cls='solid', chunk=6.0,
                  thick=1.2, h=1.0, step=1.0, **extra) -> list:
    """Chunks of a barrier that follows the track `off` m past the edge on
    `side`, from s0 to s1. Each chunk is a Prop whose extra carries its two
    front-face end points (so the chunks meet exactly) and the outward unit.
    Points closer than the class clearance to ANY part of the track are
    dropped, which splits the line where it would run into the infield.

    Chunks are cut by the BARRIER LINE's own length, not the centreline's:
    round the outside of a corner the offset line is (R + w/2 + off) / R
    times longer, so on a generated 12 m hairpin a '6 m' chunk was a 23 m
    chord whose middle sagged 1.5 m toward the road -- 29.3 m from the
    edge, under the 30 m rule. The line is sampled finely enough that its
    own spacing is <= `step`, and a chunk whose chord strays more than
    SAG_MAX from the samples it spans is split (the samples all clear the
    rule with PLACE_MARGIN to spare; the chord then clears it too)."""
    tr = ctx.tr
    if s1 <= s0:
        return []
    # the offset line's stretch over the centreline: 1 + |kappa| (w/2 + off)
    n = _nper(tr)
    kmax = float(np.abs(np.asarray(tr.kappa[:n])).max()) if n else 0.0
    stretch = 1.0 + kmax * (ctx.hw + off + thick)
    ds_ = step / stretch
    s = np.arange(s0, s1 + 1e-9, ds_)
    P, T = _at(tr, s, side * (ctx.hw + off))
    Pb, _T = _at(tr, s, side * (ctx.hw + off + thick))
    need = CLEAR[cls] + PLACE_MARGIN
    # the FRONT line faces this stretch of road; the BACK line may face
    # another (the infield): both must clear
    ok = (ctx.edge.fast(P) >= need) & (ctx.edge.fast(Pb) >= need)
    bad = np.flatnonzero(~ok)
    if len(bad):
        ok[bad] = ((ctx.edge.exact(P[bad], cap=need + 1.0) >= need)
                   & (ctx.edge.exact(Pb[bad], cap=need + 1.0) >= need))
    if ctx.edge.areas:
        ok &= np.minimum(ctx.edge.area(P), ctx.edge.area(Pb)) >= CLEAR_AREA + PLACE_MARGIN
    out = []
    for a, ln in _runs(ok, False):
        if ln < 2:
            continue
        idx = np.arange(a, a + ln)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P[idx], axis=0).T))])
        total = float(cum[-1])
        if total < 0.3:
            continue
        n_ch = max(1, int(math.ceil(total / chunk - 1e-9)))
        cuts = np.unique(np.searchsorted(cum, np.linspace(0.0, total, n_ch + 1)))
        cuts = np.clip(cuts, 0, ln - 1)
        bounds = []
        stack = [(int(c0), int(c1)) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 > c0][::-1]
        while stack:
            c0, c1 = stack.pop()
            if c1 - c0 >= 2:
                A, Bp = P[idx[c0]], P[idx[c1]]
                d = Bp - A
                L = max(float(np.hypot(*d)), 1e-9)
                M = P[idx[c0 + 1:c1]] - A
                if float(np.abs(M[:, 0] * d[1] - M[:, 1] * d[0]).max()) / L > SAG_MAX:
                    cm = (c0 + c1) // 2
                    stack += [(cm, c1), (c0, cm)]
                    continue
            bounds.append((c0, c1))
        for c0, c1 in bounds:
            i0, i1 = idx[c0], idx[c1]
            p0, p1 = P[i0], P[i1]
            d = p1 - p0
            L = float(np.hypot(*d))
            if L < 0.2:
                continue
            t = d / L
            nrm = np.array([-t[1], t[0]])
            o = nrm if float(nrm @ (side * np.array([-T[i0, 1], T[i0, 0]]))) > 0 else -nrm
            c = 0.5 * (p0 + p1) + o * 0.5 * thick
            tt = t if _cross2(t, o) > 0 else -t
            pr = Prop(kind, cls, c[0], c[1], tt, 0.5 * L, 0.5 * thick, h,
                      p0=(float(p0[0]), float(p0[1])), p1=(float(p1[0]), float(p1[1])),
                      out=(float(o[0]), float(o[1])), s=float(s[i0]), side=float(side),
                      **extra)
            # consecutive chunks SHARE an end point, so the overlap test runs
            # on footprints shrunk by 0.3 m (a grown one drops every other)
            if ctx.free(pr.foot, -0.3):
                ctx.props.append(pr)
                ctx.keep.append((pr.foot, 5.0))
                out.append(pr)
    return out


def _trees(ctx, d_min=38.0, d_max=330.0, spacing=10.5, density=1.0,
           species_bias=None, rings=None) -> None:
    """Scatter trees over the land round the track: a jittered grid thinned
    by a smooth value-noise field (clusters and clearings, not a lawn of
    dots), a falloff with distance from the edge, the keep-outs round every
    placed prop, and the clearance rule on each canopy (its trunk stands
    CLEAR_SOLID + canopy radius from the edge). Species come from two further
    noise fields, so conifers grow in stands and broadleaves in groves."""
    tr, rng, E = ctx.tr, ctx.rng, ctx.edge
    x0, y0, x1, y1 = tr.bbox
    for poly in E.areas:
        x0, y0 = min(x0, poly[:, 0].min()), min(y0, poly[:, 1].min())
        x1, y1 = max(x1, poly[:, 0].max()), max(y1, poly[:, 1].max())
    for cx, cy, R in E.rings:
        x0, y0, x1, y1 = min(x0, cx - R), min(y0, cy - R), max(x1, cx + R), max(y1, cy + R)
    pad = d_max + E.hw + 10.0
    xs = np.arange(x0 - pad, x1 + pad, spacing)
    ys = np.arange(y0 - pad, y1 + pad, spacing)
    gx, gy = np.meshgrid(xs, ys)
    P = np.column_stack([gx.ravel(), gy.ravel()])
    P += rng.uniform(-0.45, 0.45, P.shape) * spacing

    def noise(Q, cell, seed_shift):
        g = np.random.default_rng(zlib.crc32((ctx.name + str(seed_shift)).encode()))
        nx = int((x1 - x0 + 2 * pad) / cell) + 3
        ny = int((y1 - y0 + 2 * pad) / cell) + 3
        V = g.random((ny, nx))
        fx = (Q[:, 0] - (x0 - pad)) / cell
        fy = (Q[:, 1] - (y0 - pad)) / cell
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        u, v = fx - i, fy - j
        u, v = u * u * (3 - 2 * u), v * v * (3 - 2 * v)
        i, j = np.clip(i, 0, nx - 2), np.clip(j, 0, ny - 2)
        return ((V[j, i] * (1 - u) + V[j, i + 1] * u) * (1 - v)
                + (V[j + 1, i] * (1 - u) + V[j + 1, i + 1] * u) * v)

    e = E.fast(P)
    keep = (e > d_min - 4.0) & (e < d_max)
    if E.areas:
        keep &= E.area(P) > CLEAR_AREA + 14.0
    P, e = P[keep], e[keep]
    n1 = noise(P, 62.0, 'a')
    n2 = noise(P, 23.0, 'b')
    # full density near the track, thinning with distance: the far belts are
    # seen as a skyline at 200-430 m, where a tree is 30-60 px and every
    # sprite is a blit -- measured, the arena's start frame drew 346 trees
    # with 311 of them past 200 m. So the far land is thinner, and mostly
    # COPSES (one sprite the width of four trees), which keeps the skyline
    # full for a third of the blits.
    fall = np.clip((e - d_min + 4.0) / 16.0, 0.0, 1.0) * np.interp(
        e, [0.0, 95.0, 160.0, 240.0, 1e9], [1.0, 1.0, 0.52, 0.34, 0.30])
    p_acc = np.clip(1.9 * (0.62 * n1 + 0.38 * n2) - 0.62, 0.0, 1.0) * fall * density
    acc = rng.random(len(P)) < p_acc
    P, e = P[acc], e[acc]
    ns = noise(P, 90.0, 'c')
    nb = noise(P, 35.0, 'd')
    rr = rng.random(len(P))
    sp = np.where(ns > 0.62, 1, 0)                       # conifer stands
    sp = np.where((ns < 0.30) & (rr < 0.35), 3, sp)       # copses
    sp = np.where((nb > 0.78) & (sp == 0), 2, sp)         # poplars
    sp = np.where(rr > 0.93, 4, sp)                       # bushes
    far = (e > 150.0) & (rng.random(len(P)) < np.clip((e - 150.0) / 120.0, 0.0, 0.62))
    sp = np.where(far & (sp != 1), 3, sp)                 # far land: copses
    if species_bias is not None:
        sp = species_bias(P, sp, rng)
    H = np.empty(len(P))
    Rc = np.empty(len(P))
    for k, name in enumerate(SPECIES_ORDER):
        m = sp == k
        lo, hi = SPECIES[name]['h']
        H[m] = lo + (hi - lo) * rng.random(int(m.sum())) ** 0.8
        Rc[m] = SPECIES[name]['r'] * H[m]
    ok = e >= CLEAR_SOLID + Rc + PLACE_MARGIN
    for poly, margin in ctx.keep:
        c = poly.mean(axis=0)
        r = float(np.hypot(*(poly - c).T).max())
        near = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < r + margin + Rc
        if near.any():
            idx = np.flatnonzero(near)
            grown = _rect_pts(c[0], c[1], _axis(poly), 0.5 * _len(poly, 0) + margin,
                              0.5 * _len(poly, 1) + margin)
            inside = _pip(P[idx], grown)
            d = _seg_dist(P[idx], grown, np.roll(grown, -1, axis=0))
            ok[idx[inside | (d < Rc[idx])]] = False
    P, H, Rc, sp = P[ok], H[ok], Rc[ok], sp[ok]
    # the exact clearance, paid only for the survivors
    if len(P):
        ex = E.exact(P, cap=CLEAR_SOLID + float(Rc.max()) + 2.0)
        good = ex >= CLEAR_SOLID + Rc + PLACE_MARGIN
        if E.areas:
            good &= E.area(P) >= CLEAR_AREA + Rc + PLACE_MARGIN
        P, H, Rc, sp = P[good], H[good], Rc[good], sp[good]
    shape = np.array([rng.integers(0, SHAPES[SPECIES_ORDER[k]]) for k in sp],
                     dtype=np.int64)
    ctx.trees.append((P, H, Rc, sp, shape))


def _axis(poly) -> tuple:
    d = poly[1] - poly[0]
    n = math.hypot(*d) or 1.0
    return (d[0] / n, d[1] / n)


def _len(poly, k) -> float:
    return float(math.hypot(*(poly[1 + k] - poly[k])))


def _off(cls: str, half_across: float) -> float:
    """How far past the edge to put the CENTRE of a prop whose footprint
    reaches `half_across` m toward the road: the class minimum, the placement
    margin and 0.2 m to spare."""
    return CLEAR[cls] + PLACE_MARGIN + half_across + 0.2


def _put_xy(ctx, kind, cls, P, t, hl, hw, h, gap=1.0, keep=4.0, **extra):
    """Place a prop at a world point with an explicit axis t (front = -o)."""
    p = Prop(kind, cls, P[0], P[1], t, hl, hw, h, **extra)
    return p if ctx.place(p, gap, keep) else None


def _facing(t_track) -> tuple:
    """The axis t of a prop that FACES oncoming cars: its o (t rotated +90
    deg) points down the road, so its front (-o) looks back up it."""
    return (float(t_track[1]), -float(t_track[0]))


def _slow(c) -> bool:
    """Braked for, with a stand: R <= R_SLOW, or a half-turn (task 46)."""
    return c['R'] <= R_SLOW or c.get('turn', 0.0) >= TURN_HALF_DEG


def _real(c) -> bool:
    """Ends a braking zone, walled with tyres: R <= R_KINK, or a half-turn."""
    return c['R'] <= R_KINK or c.get('turn', 0.0) >= TURN_HALF_DEG


def _brake_boards(ctx, corners) -> None:
    """300 / 200 / 100 m boards (3 / 2 / 1 diagonal stripes) before every
    SLOW corner, on its outside (the side the car brakes on), on the approach
    straight: a kink of R > R_KINK does not end the approach, a real corner
    does, and a board that would stand in the previous corner is not placed."""
    tr = ctx.tr
    L = float(tr.length)
    real = [c for c in corners if _real(c)]
    for c in corners:
        if not _slow(c):
            continue
        prev = [q for q in real if q is not c]
        if tr.closed and prev:
            gaps = [((c['s_in'] - q['s_out']) % L) for q in prev]
            approach = min(gaps)
        else:
            ends = [c['s_in'] - q['s_out'] for q in prev if q['s_out'] <= c['s_in']]
            approach = min(ends) if ends else c['s_in']
        side = -c['sgn']
        for n_str, d in ((3, 300.0), (2, 200.0), (1, 100.0)):
            if d > approach - 8.0:
                continue
            s = c['s_in'] - d
            P, t, o = ctx.frame(s, side, _off('board', 0.9))
            _put_xy(ctx, 'brake', 'board', P, _facing(t), 0.9, 0.12, 2.9,
                    gap=0.5, keep=3.0, stripes=n_str, s=float(s))


def _theme_circuit(ctx) -> None:
    """A permanent circuit: see the module docstring. Order is priority: the
    gantry, the pits, the stands, then the barriers, boards, posts, lights,
    and last the trees, which fill whatever land is left."""
    tr, rng = ctx.tr, ctx.rng
    L = float(tr.length)
    hw = ctx.hw
    ins = ctx.inside
    outs = -ins
    corners = _corners(tr)
    straights = _straights(tr)
    # --- the main straight: the one the start line is on, else the longest
    main = None
    for s0, s1 in straights:
        if s0 - 2.0 <= 0.0 <= s1 + 2.0 or (tr.closed and s0 - 2.0 <= L <= s1 + 2.0):
            main = (s0, s1)
    if main is None and straights:
        main = max(straights, key=lambda r: r[1] - r[0])
    if main is None:
        main = (0.0, min(120.0, L))
    s_a, s_b = main
    if tr.closed and s_a > 0.5 * L:          # a straight THROUGH the seam
        s_a -= L
        s_b -= L
    Lm = s_b - s_a

    # --- the start / finish gantry: two pillars CLEAR_GANTRY past the edges
    P0, t0, _o = ctx.frame(0.0, 1.0, _off('gantry', 0.45))
    P1, _t, _o = ctx.frame(0.0, -1.0, _off('gantry', 0.45))
    pa = _put_xy(ctx, 'pillar', 'gantry', P0, t0, 0.45, 0.45, 7.3, gap=0.3, keep=3.0)
    pb = _put_xy(ctx, 'pillar', 'gantry', P1, t0, 0.45, 0.45, 7.3, gap=0.3, keep=3.0)
    if pa is not None and pb is not None:
        mid = 0.5 * (P0 + P1)
        span = float(np.hypot(*(P1 - P0)))
        g = Prop('gantry', 'overhead', mid[0], mid[1], (P1 - P0) / span,
                 0.5 * span, 0.5, 7.4, a=(float(P0[0]), float(P0[1])),
                 b=(float(P1[0]), float(P1[1])), road=(float(t0[0]), float(t0[1])))
        ctx.props.append(g)

    # --- pits + control tower on the infield side, main stand outside
    pits = None
    for side in (ins, outs):
        for frac in (0.62, 0.5, 0.4, 0.3):
            Lp = min(110.0, frac * Lm)
            if Lp < 36.0:
                continue
            sc = s_a + 0.12 * Lm + 0.5 * Lp
            pits = ctx.put('pits', 'solid', sc, side, 33.5, Lp, 12.0, 8.0,
                           tries=(0.0, 2.0, 5.0), keep=8.0)
            if pits is not None:
                break
        if pits is not None:
            break
    if pits is not None:
        sd = pits.extra['side']
        for s_t in (pits.extra['s'] - pits.hl - 8.0, pits.extra['s'] + pits.hl + 8.0):
            if ctx.put('tower', 'solid', s_t, sd, 34.0, 9.0, 9.0, 21.0,
                       tries=(0.0, 3.0, 8.0), keep=8.0) is not None:
                break
        # paddock: trailers in a row behind the building
        n_tr = int(2 * pits.hl // 17.0)
        for k in range(n_tr):
            s_k = pits.extra['s'] - pits.hl + 8.5 + 17.0 * k
            ctx.put('trailer', 'solid', s_k, sd, 33.5 + 12.0 + 4.0, 13.0, 2.6,
                    3.9, tries=(0.0, 3.0), keep=3.0, tone=int(rng.integers(0, 6)))
        for k in range(3):
            s_k = pits.extra['s'] - pits.hl + (k + 0.5) * 2 * pits.hl / 3.0
            ctx.put('flood', 'solid', s_k, sd, 33.5 + 12.0 + 12.0, 3.4, 1.2,
                    24.0, tries=(0.0, 4.0, 9.0), keep=3.0)
        s_f = pits.extra['s'] + pits.hl + 8.0
        for k in range(6):
            ctx.put('flag', 'solid', s_f + 3.6 * k, sd, 35.0, 0.4, 0.4, 11.0,
                    tries=(0.0, 3.0), keep=2.0, tone=k % len(FLAGS))
    stand_side = outs if (pits is None or pits.extra['side'] == ins) else ins
    stand = None
    for frac in (0.6, 0.48, 0.36):
        Ls = min(96.0, frac * Lm)
        if Ls < 30.0:
            continue
        stand = ctx.put('stand', 'solid', s_a + 0.5 * Lm + 0.06 * Lm, stand_side,
                        36.0, Ls, 13.0, 14.0, tries=(0.0, 4.0, 9.0), keep=10.0,
                        roof=True, livery=0)
        if stand is not None:
            break
    if stand is not None:
        for k in range(4):
            s_k = stand.extra['s'] - stand.hl + (k + 0.5) * 2 * stand.hl / 4.0
            ctx.put('flood', 'solid', s_k, stand_side, 36.0 + 13.0 + 7.0, 3.4,
                    1.2, 25.0, tries=(0.0, 4.0), keep=3.0)
    # --- a stand at the slowest corner, on its outside
    slow = sorted([c for c in corners if _slow(c)], key=lambda c: c['R'])
    for c in slow[:2]:
        st2 = ctx.put('stand', 'solid', c['s_mid'], -c['sgn'], 36.0, 44.0, 11.0,
                      12.5, tries=(0.0, 4.0, 9.0, 16.0), keep=10.0, roof=True, livery=1)
        if st2 is not None:
            break

    # --- barriers at the run-off boundary: tyre walls on the outside of
    #     every real corner (and 30 m either side), armco elsewhere
    n = int(math.ceil(L))
    sgrid = np.arange(n, dtype=np.float64)
    for side in (1.0, -1.0):
        kind = np.array(['armco'] * n, dtype=object)
        for c in corners:
            if not _real(c) or -c['sgn'] != side:
                continue
            a0, a1 = c['s_in'] - 30.0, c['s_out'] + 40.0
            if tr.closed:
                m = ((sgrid - a0) % L) <= (a1 - a0)
            else:
                m = (sgrid >= a0) & (sgrid <= a1)
            kind[m] = 'tyres'
        # contiguous runs of one kind (the lap seam joins its two ends)
        edges = np.flatnonzero(kind[1:] != kind[:-1]) + 1
        bounds = [0] + edges.tolist() + [n]
        runs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        if tr.closed and len(runs) > 1 and kind[0] == kind[-1]:
            a, b = runs.pop()
            runs[0] = (a - n, runs[0][1])
        for a, b in runs:
            kk = kind[a % n]
            if kk == 'tyres':
                lines = _barrier_line(ctx, float(a), float(b), side, 30.8, 'tyres',
                                      chunk=8.0, thick=1.3, h=1.05)
            else:
                lines = _barrier_line(ctx, float(a), float(b), side, 31.0, 'armco',
                                      chunk=10.0, thick=0.5, h=0.82)
            for j, pr in enumerate(lines):
                pr.extra['belt'] = int(j % 2)
        # sponsor boards behind the barrier, every ~34 m where it fits
        for s in np.arange(17.0 + 11.0 * (side > 0), L, 34.0):
            ctx.put('board', 'solid', float(s), side, 33.6, 6.0, 0.35, 2.5,
                    tries=(0.0, 1.0), gap=0.3, keep=2.5,
                    design=int(rng.integers(0, len(DESIGNS))))
    # --- a catch fence in front of each stand, behind the barrier
    for p in [q for q in ctx.props if q.kind == 'stand']:
        s0 = p.extra['s'] - p.hl - 12.0
        _barrier_line(ctx, s0, s0 + 2 * p.hl + 24.0, p.extra['side'], 32.4,
                      'fence', chunk=12.0, thick=0.2, h=4.2)
    _brake_boards(ctx, corners)
    # --- marshal posts round the lap, alternating sides
    nm = max(3, int(L // 190.0))
    for k in range(nm):
        s = 60.0 + k * L / nm
        side = 1.0 if k % 2 == 0 else -1.0
        if ctx.put('marshal', 'solid', s, side, 34.2, 2.4, 2.2, 2.6,
                   tries=(0.0, 1.5), keep=3.0) is None:
            ctx.put('marshal', 'solid', s, -side, 34.2, 2.4, 2.2, 2.6,
                    tries=(0.0, 1.5), keep=3.0)
    # --- a few more floodlights round the lap
    for k in range(4):
        s = (k + 0.5) * L / 4.0 + 40.0
        ctx.put('flood', 'solid', s, outs, 44.0, 3.4, 1.2, 24.0,
                tries=(0.0, 6.0, 14.0), keep=3.0)
    _trees(ctx, d_min=38.0, d_max=330.0, spacing=10.5)


def _rrect_pts(cx, cy, hx, hy, r, step=1.0):
    """Outline of a rounded rectangle, counter-clockwise, every ~step m, and
    the outward unit normal at each point."""
    P, N = [], []
    sx, sy = hx - r, hy - r
    for (ccx, ccy, a0, lx, ly, dx, dy) in (
            (cx + sx, cy - sy, -0.5 * math.pi, 2 * sy, 0.0, 0.0, 1.0),
            (cx + sx, cy + sy, 0.0, 2 * sx, 0.0, -1.0, 0.0),
            (cx - sx, cy + sy, 0.5 * math.pi, 2 * sy, 0.0, 0.0, -1.0),
            (cx - sx, cy - sy, math.pi, 2 * sx, 0.0, 1.0, 0.0)):
        n_a = max(2, int(0.5 * math.pi * r / step))
        for k in range(n_a):
            a = a0 + 0.5 * math.pi * k / n_a
            P.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
            N.append((math.cos(a), math.sin(a)))
        a = a0 + 0.5 * math.pi
        ex, ey = ccx + r * math.cos(a), ccy + r * math.sin(a)
        ln = max(lx, ly)
        n_s = max(1, int(ln / step))
        for k in range(n_s):
            P.append((ex + dx * ln * k / n_s, ey + dy * ln * k / n_s))
            N.append((math.cos(a), math.sin(a)))
    return np.array(P), np.array(N)


def _line_props(ctx, P, O, kind, cls='solid', chunk=6.0, thick=0.3, h=1.0,
                closed=True, **extra) -> list:
    """Chunks along an explicit front line P with outward normals O."""
    need = CLEAR[cls] + PLACE_MARGIN
    Pb = P + O * thick
    ok = (ctx.edge.fast(P) >= need) & (ctx.edge.fast(Pb) >= need)
    if ctx.edge.areas:
        ok &= ctx.edge.area(P) >= CLEAR_AREA + PLACE_MARGIN
    out = []
    step = float(np.median(np.hypot(*np.diff(P, axis=0).T))) if len(P) > 1 else 1.0
    k = max(2, int(round(chunk / max(step, 1e-6))))
    for a, ln in _runs(ok, closed):
        idx = (a + np.arange(ln + (1 if (closed and ln < len(P)) else 0))) % len(P)
        for j in range(0, len(idx) - 1, k):
            seg = idx[j:j + k + 1]
            if len(seg) < 2:
                continue
            p0, p1 = P[seg[0]], P[seg[-1]]
            d = p1 - p0
            Lc = float(np.hypot(*d))
            if Lc < 0.3 * chunk:
                continue
            t = d / Lc
            nrm = np.array([-t[1], t[0]])
            o = nrm if float(nrm @ O[seg[0]]) > 0 else -nrm
            tt = t if _cross2(t, o) > 0 else -t
            c = 0.5 * (p0 + p1) + o * 0.5 * thick
            pr = Prop(kind, cls, c[0], c[1], tt, 0.5 * Lc, 0.5 * thick, h,
                      p0=(float(p0[0]), float(p0[1])), p1=(float(p1[0]), float(p1[1])),
                      out=(float(o[0]), float(o[1])), **extra)
            if ctx.free(pr.foot, -0.3):
                ctx.props.append(pr)
                ctx.keep.append((pr.foot, 5.0))
                out.append(pr)
    return out


def _theme_proving(ctx) -> None:
    """The open map: a test centre round a 522 x 362 m tarmac pad. Hangars
    and a tower to the north, an office to the west, a windsock, a chain-link
    perimeter fence 27 m out (a gate in front of the hangars), light masts,
    parked cone stacks, and trees beyond the fence."""
    tr, rng = ctx.tr, ctx.rng
    pad = None
    for a in getattr(tr, 'areas', []) or []:
        if a.drivable and a.kind == 'rrect':
            pad = a
            break
    if pad is None:
        _theme_circuit(ctx)
        return
    # The land plan is laid out round the UNION of the pad and the road: the
    # open map's perimeter road is not centred in its pad (the north straight
    # runs 33 m outside it), so the pad alone would put a hangar on the road.
    _pcx, _pcy, _phx, _phy, r = pad.params
    bx0, by0, bx1, by1 = tr.bbox
    x0 = min(_pcx - _phx, bx0 - ctx.hw)
    x1 = max(_pcx + _phx, bx1 + ctx.hw)
    y0 = min(_pcy - _phy, by0 - ctx.hw)
    y1 = max(_pcy + _phy, by1 + ctx.hw)
    cx, cy, hx, hy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    # hangars along the north side, doors facing the pad (south)
    for k, dx in enumerate((-150.0, 0.0, 150.0)):
        P = np.array([cx + dx, y1 + 46.0 + 22.0])
        _put_xy(ctx, 'hangar', 'solid', P, (1.0, 0.0), 28.0, 22.0, 16.5,
                gap=2.0, keep=10.0, tone=k)
    _put_xy(ctx, 'tower', 'solid', np.array([cx + 245.0, y1 + 62.0]), (1.0, 0.0),
            4.0, 4.0, 27.0, keep=8.0, slim=True)
    _put_xy(ctx, 'office', 'solid', np.array([x0 - 52.0, cy + 40.0]), (0.0, -1.0),
            22.0, 7.0, 7.5, keep=8.0)
    _put_xy(ctx, 'office', 'solid', np.array([x0 - 50.0, cy - 30.0]), (0.0, -1.0),
            12.0, 6.0, 5.0, keep=8.0)
    _put_xy(ctx, 'windsock', 'solid', np.array([x1 + 46.0, y0 - 34.0]),
            (1.0, 0.0), 0.3, 0.3, 7.5, keep=3.0)
    for k, (px, py) in enumerate(((x0 - 44.0, cy + 95.0), (x0 - 40.0, cy + 110.0),
                                  (cx + 120.0, y0 - 42.0))):
        _put_xy(ctx, 'cones', 'solid', np.array([px, py]), (1.0, 0.0), 2.2, 1.2, 0.9,
                keep=2.0, n=int(rng.integers(4, 7)))
    for sx_, sy_ in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        P = np.array([cx + sx_ * (hx + 40.0), cy + sy_ * (hy + 40.0)])
        o = -np.array([sx_, sy_], dtype=float) / math.sqrt(2.0)
        _put_xy(ctx, 'flood', 'solid', P, (o[1], -o[0]), 1.7, 0.6, 26.0, keep=3.0)
    # a workshop row and its car park south of the pad, light masts along it
    _put_xy(ctx, 'pits', 'solid', np.array([x0 + 118.0, y0 - 54.0]), (-1.0, 0.0), 36.0, 8.0,
            6.2, keep=8.0, shed=True)
    for row, yy in enumerate((y0 - 44.0, y0 - 51.5)):
        for k in range(9):
            if rng.random() < 0.22:
                continue                          # an empty bay
            _put_xy(ctx, 'car', 'solid', np.array([x0 + 8.0 + 3.1 * k, yy]),
                    (0.0, 1.0 if row == 0 else -1.0), 2.1, 0.9, 1.45, gap=0.3, keep=2.0,
                    tone=int(rng.integers(0, 7)))
    for dx in (-150.0, 0.0, 150.0):
        P = np.array([cx + dx + 40.0, y0 - 40.0])
        _put_xy(ctx, 'flood', 'solid', P, (-1.0, 0.0), 1.7, 0.6, 22.0, keep=3.0)
    # the perimeter fence, with a gate in front of the middle hangar
    g = CLEAR_SOLID + PLACE_MARGIN + 1.0
    P, N = _rrect_pts(cx, cy, hx + g, hy + g, r + g, step=1.0)
    gate = (np.abs(P[:, 0] - cx) < 34.0) & (P[:, 1] > cy)
    _line_props(ctx, P[~gate], N[~gate], 'fence', chunk=24.0, thick=0.2, h=2.6,
                closed=False)
    _trees(ctx, d_min=46.0, d_max=320.0, spacing=11.0, density=0.95)


def _theme_skidpad(ctx) -> None:
    """A skidpad: the guide circles are road (they are driven), so every prop
    is 30 m past the OUTERMOST circle's edge. A timing hut facing the pad,
    four light masts, a windsock, cone stacks, trees."""
    tr, rng = ctx.tr, ctx.rng
    cen = getattr(tr, 'centre', ()) or ()
    if len(cen) != 2:
        _theme_circuit(ctx)
        return
    cx, cy = float(cen[0]), float(cen[1])
    # the outermost driven radius: the centreline circle or a guide circle
    R_env = max([float(np.hypot(*(tr.xy[0] - np.array(cen))))]
                + list(getattr(tr, 'guide_radii', []) or [])) + ctx.hw

    def at(bearing, dist):
        return np.array([cx + dist * math.cos(bearing), cy + dist * math.sin(bearing)])

    b = -0.5 * math.pi
    o = np.array([math.cos(b), math.sin(b)])        # outward from the pad
    _put_xy(ctx, 'office', 'solid', at(b, R_env + 36.0 + 5.0), (o[1], -o[0]), 5.5, 4.0,
            4.2, keep=8.0, hut=True)
    _put_xy(ctx, 'cones', 'solid', at(b + 0.09, R_env + 38.0), (o[1], -o[0]), 2.2, 1.2, 0.9,
            keep=2.0, n=5)
    for k in range(4):
        bb = 0.25 * math.pi + 0.5 * math.pi * k
        o = np.array([math.cos(bb), math.sin(bb)])
        _put_xy(ctx, 'flood', 'solid', at(bb, R_env + 38.0), (o[1], -o[0]), 1.7, 0.6,
                24.0, keep=3.0)
    _put_xy(ctx, 'windsock', 'solid', at(2.6, R_env + 42.0), (1.0, 0.0), 0.3, 0.3, 7.5,
            keep=3.0)
    _trees(ctx, d_min=40.0, d_max=300.0, spacing=10.5)


def _theme_drag(ctx) -> None:
    """A drag strip: low concrete walls CLEAR_WALL past both edges, the
    start-light tree 30 m down the strip (in view past the HUD at the
    launch), a timing board and gate posts at
    every `Track.gates` line, bleachers and a timing tower at the start, light
    poles down both sides, trees."""
    tr, rng = ctx.tr, ctx.rng
    L = float(tr.length)
    hw = ctx.hw
    for side in (1.0, -1.0):
        _barrier_line(ctx, -60.0, L, side, CLEAR_WALL + PLACE_MARGIN + 0.2, 'wall',
                      cls='wall',
                      chunk=20.0, thick=0.6, h=1.05, step=2.0)
    # the start-light tree: beside the strip 30 m past the launch line,
    # facing the car on it. The chase eye sits ~6 m behind a staged car, so
    # the tree is (s + 6) m deep and 11.65 m to the side: at s = 10 that was
    # 35 deg off the axis, just outside the 34 deg half-frame; at 14 (29 deg,
    # x 78-121 px of 1280) it stood under the full HUD's left telemetry panel;
    # at 22-26 (the panel's 'mu' row runs past its box to x 285) still partly.
    # At 30 (18 deg) the lamp housing clears every HUD pixel by >= 12 px at
    # 1024x640, 1280x800, 1440x900 and 1920x1080 while the car is staged, and
    # the launch frame (car at s = 6) shows it too
    P, t, o = ctx.frame(30.0, 1.0, _off('xmas', 0.35))
    _put_xy(ctx, 'xmas', 'xmas', P, _facing(t), 0.35, 0.3, 4.0, gap=0.3, keep=2.0)
    for s_g, label in (getattr(tr, 'gates', []) or []):
        if s_g <= 1.0:
            continue
        # the scoreboard stands BEHIND the wall, the timing eyes inside it
        P, t, o = ctx.frame(s_g, -1.0, CLEAR_WALL + PLACE_MARGIN + 0.8 + 3.2)
        _put_xy(ctx, 'timing', 'board', P, _facing(t), 2.3, 0.2, 5.4, gap=0.3,
                keep=3.0, s=float(s_g))
        for side in (1.0, -1.0):
            P, t, o = ctx.frame(s_g, side, _off('board', 0.2))
            _put_xy(ctx, 'gatepost', 'board', P, _facing(t), 0.2, 0.2, 5.0, gap=0.2, keep=2.0)
    for side in (1.0, -1.0):
        ctx.put('stand', 'solid', 5.0, side, 36.0, 84.0, 9.0, 5.6,
                tries=(0.0, 4.0), keep=8.0, roof=False, livery=2)
    ctx.put('tower', 'solid', -40.0, 1.0, 40.0, 7.0, 7.0, 12.0,
            tries=(0.0, 4.0, 10.0), keep=6.0)
    for s in np.arange(60.0, L + 1.0, 150.0):
        for side in (1.0, -1.0):
            P, t, o = ctx.frame(float(s), side, 33.0)
            _put_xy(ctx, 'lightpole', 'solid', P, _facing(t), 0.3, 0.3, 16.0,
                    gap=0.5, keep=2.0)

    def poplar_rows(P, sp, g):
        # a poplar windbreak along both sides, 70-80 m out
        n = np.abs(P[:, 1] - float(tr.xy[0, 1]))
        row = (n > 70.0) & (n < 82.0) & (g.random(len(P)) < 0.8)
        return np.where(row, 2, sp)
    _trees(ctx, d_min=40.0, d_max=260.0, spacing=11.0, density=0.8,
           species_bias=poplar_rows)


THEMES = {'arena': _theme_circuit, 'linden': _theme_circuit, 'kestrel': _theme_circuit,
          'ashdown': _theme_circuit, 'fairfield': _theme_circuit, 'open': _theme_proving,
          'skidpad': _theme_skidpad, 'dragstrip': _theme_drag}


class Layout:
    """The props and trees of one track: `props` (list of Prop), and the
    trees as arrays xy (T, 2), h, r (canopy radius), sp (species index),
    shape, detail (True = drawn at detail 'high' only)."""

    def __init__(self, tr):
        t0 = time.perf_counter()
        ctx = _Ctx(tr)
        theme = THEMES.get(ctx.name, _theme_circuit)
        theme(ctx)
        self.tr = tr
        self.name = ctx.name
        self.theme = theme.__name__.replace('_theme_', '')
        self.props = ctx.props
        if ctx.trees:
            self.xy = np.vstack([b[0] for b in ctx.trees])
            self.h = np.concatenate([b[1] for b in ctx.trees])
            self.r = np.concatenate([b[2] for b in ctx.trees])
            self.sp = np.concatenate([b[3] for b in ctx.trees]).astype(np.int64)
            self.shape = np.concatenate([b[4] for b in ctx.trees]).astype(np.int64)
        else:
            self.xy = np.zeros((0, 2))
            self.h = self.r = np.zeros(0)
            self.sp = self.shape = np.zeros(0, dtype=np.int64)
        # half the trees are 'detail' (dropped at cfg.detail 'low'): every
        # other one in a seeded shuffle, so a low-detail wood is thinner, not
        # missing a corner
        g = np.random.default_rng(zlib.crc32((self.name + 'lod').encode()))
        self.detail = g.random(len(self.xy)) < 0.5
        self.build_ms = (time.perf_counter() - t0) * 1e3

    def digest(self) -> str:
        """A hash of everything placed: the determinism check."""
        h = zlib.crc32(repr([p.key() for p in self.props]).encode())
        for a in (self.xy, self.h, self.r):
            h = zlib.crc32(np.round(a, 3).tobytes(), h)
        h = zlib.crc32(self.sp.tobytes(), h)
        h = zlib.crc32(self.shape.tobytes(), h)
        return f'{h:08x}'

    def counts(self) -> dict:
        out = {}
        for p in self.props:
            out[p.kind] = out.get(p.kind, 0) + 1
        out['tree'] = int(len(self.xy))
        return out


# ======================================================================= #
#  GEOMETRY: one world-space soup for the chase view, footprints for plan #
# ======================================================================= #
INF = float('inf')
_Z = np.array([0.0, 0.0, 1.0])


def _newell(V) -> np.ndarray:
    """Unit normal of a (possibly slightly non-planar) polygon, by Newell's
    method: robust where three chosen vertices would be nearly collinear."""
    V = np.asarray(V, dtype=np.float64)
    W = np.roll(V, -1, axis=0)
    n = np.array([((V[:, 1] - W[:, 1]) * (V[:, 2] + W[:, 2])).sum(),
                  ((V[:, 2] - W[:, 2]) * (V[:, 0] + W[:, 0])).sum(),
                  ((V[:, 0] - W[:, 0]) * (V[:, 1] + W[:, 1])).sum()])
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 1e-12 else np.zeros(3)


class _Soup:
    """Builder: polygons (kind 0 opaque, 1 line, 2 alpha ground), their
    objects, and the plan items. `finish()` flattens everything into arrays."""

    def __init__(self):
        self.V = []
        self.P = []          # (obj, kind, rgb, sub, layer, lo, hi, w, two, nrm)
        self.obj_rmax = []
        self.obj_detail = []
        self.Q = []          # plan: (obj, kind, verts2, rgb, layer, z, w, minpx)
        self.obj_kind = []   # the prop kind that made each object (the checks)
        self.kind = ''

    def obj(self, rmax: float, detail: bool = False) -> int:
        self.obj_rmax.append(float(rmax))
        self.obj_detail.append(bool(detail))
        self.obj_kind.append(self.kind)
        return len(self.obj_rmax) - 1

    def face(self, V, rgb, oid, out, sub=0, lo=0.0, hi=INF, plan=True,
             two=False, shade=True):
        """A polygon, wound so its normal points along `out`, lit by the sun.
        An up-facing face is also a plan footprint (a roof, a tread, a top)."""
        V = np.asarray(V, dtype=np.float64)
        n = _newell(V)
        if float(n @ np.asarray(out, dtype=np.float64)) < 0.0:
            V, n = V[::-1].copy(), -n
        col = _shade(rgb, n) if shade else tuple(int(c) for c in rgb)
        self.V.append(V)
        self.P.append((oid, 0, col, sub, 1, lo, hi, 0.0, two, n))
        if plan and n[2] > 0.3:
            self.Q.append((oid, 0, V[:, :2].copy(), col, 2, float(V[:, 2].max()), 0.0, 0.0))

    def line(self, a, b, rgb, oid, w, lo=0.0, hi=INF, plan=True, sub=0):
        """A 3-D segment `w` m wide (a pole, a post, a wire)."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        col = tuple(int(round(c * POLE_SHADE)) for c in rgb)
        self.V.append(np.vstack([a, b]))
        self.P.append((oid, 1, col, sub, 1, lo, hi, float(w), True, np.zeros(3)))
        if not plan:
            return
        if abs(b[2] - a[2]) > math.hypot(b[0] - a[0], b[1] - a[1]):
            self.Q.append((oid, 2, a[None, :2].copy(), col, 4, float(max(a[2], b[2])),
                           0.5 * float(w), 1.5))
        else:
            self.Q.append((oid, 1, np.vstack([a[:2], b[:2]]), col, 1,
                           float(max(a[2], b[2])), float(w), 1.0))

    def ground(self, poly2, oid, alpha=False, rgb=SHADOW_RGB):
        """A ground polygon (a cast shadow): layer 0, drawn before anything
        upright. alpha=True darkens whatever is under it (road, run-off)."""
        poly2 = np.asarray(poly2, dtype=np.float64)
        if len(poly2) < 3:
            return
        V = np.column_stack([poly2, np.full(len(poly2), 0.02)])
        if _newell(V)[2] < 0.0:
            V = V[::-1].copy()
        col = tuple(int(c) for c in rgb)
        self.V.append(V)
        self.P.append((oid, 2 if alpha else 0, col, 0, 0, 0.0, INF, 0.0, True, _Z.copy()))
        self.Q.append((oid, 3 if alpha else 0, V[:, :2].copy(), col, 0, 0.0, 0.0, 0.0))

    def pline(self, a2, b2, rgb, oid, w, minpx=2.0, layer=1, z=0.0):
        """A plan-only stroke (a barrier's colour, a board)."""
        self.Q.append((oid, 1, np.vstack([np.asarray(a2, float), np.asarray(b2, float)]),
                       tuple(int(c) for c in rgb), layer, float(z), float(w), float(minpx)))

    # ------------------------------------------------------------------ #
    def finish(self):
        """Flatten into the arrays the frame reads."""
        g = _Geo()
        nP = len(self.P)
        cnt = np.array([len(v) for v in self.V], dtype=np.int64)
        g.V = np.vstack(self.V) if self.V else np.zeros((0, 3))
        g.p_cnt = cnt
        g.p_st = np.cumsum(cnt) - cnt
        g.p_obj = np.array([p[0] for p in self.P], dtype=np.int64)
        g.p_kind = np.array([p[1] for p in self.P], dtype=np.int64)
        g.p_col = np.array([p[2] for p in self.P], dtype=np.float64).reshape(nP, 3)
        g.p_sub = np.array([p[3] for p in self.P], dtype=np.int64)
        g.p_layer = np.array([p[4] for p in self.P], dtype=np.int64)
        g.p_lo = np.array([p[5] for p in self.P], dtype=np.float64)
        g.p_hi = np.array([p[6] for p in self.P], dtype=np.float64)
        g.p_lo_l = g.p_lo * DETAIL_LOW_R
        g.p_hi_l = g.p_hi * DETAIL_LOW_R
        g.p_w = np.array([p[7] for p in self.P], dtype=np.float64)
        g.p_two = np.array([p[8] for p in self.P], dtype=bool)
        g.p_nrm = np.array([p[9] for p in self.P], dtype=np.float64).reshape(nP, 3)
        g.p_cen = (np.add.reduceat(g.V, g.p_st, axis=0) / cnt[:, None]
                   if nP else np.zeros((0, 3)))
        nO = len(self.obj_rmax)
        g.o_rmax = np.array(self.obj_rmax, dtype=np.float64)
        g.o_detail = np.array(self.obj_detail, dtype=bool)
        lo = np.full((nO, 3), np.inf)
        hi = np.full((nO, 3), -np.inf)
        vo = np.repeat(g.p_obj, cnt)
        if len(vo):
            np.minimum.at(lo, vo, g.V)
            np.maximum.at(hi, vo, g.V)
        empty = ~np.isfinite(lo[:, 0])
        lo[empty], hi[empty] = 0.0, 0.0
        g.o_c = 0.5 * (lo + hi)
        g.o_rad = 0.5 * np.linalg.norm(hi - lo, axis=1)
        g.o_land = hi[:, 2] > LAND_Z
        # plan items, in draw order: layer, then height
        order = sorted(range(len(self.Q)), key=lambda i: (self.Q[i][4], self.Q[i][5]))
        Q = [self.Q[i] for i in order]
        qc = np.array([len(q[2]) for q in Q], dtype=np.int64)
        g.q_V = np.vstack([q[2] for q in Q]) if Q else np.zeros((0, 2))
        g.q_cnt = qc
        g.q_st = np.cumsum(qc) - qc
        g.q_obj = np.array([q[0] for q in Q], dtype=np.int64)
        g.q_kind = [int(q[1]) for q in Q]
        g.q_col = [tuple(q[3]) for q in Q]
        g.q_layer = np.array([q[4] for q in Q], dtype=np.int64)
        g.q_w = np.array([q[6] for q in Q], dtype=np.float64)
        g.q_min = np.array([q[7] for q in Q], dtype=np.float64)
        # plan cull radius per object: its own verts in plan (a shadow reaches
        # further than the solid that casts it)
        qlo = np.full((nO, 2), np.inf)
        qhi = np.full((nO, 2), -np.inf)
        qo = np.repeat(g.q_obj, qc)
        if len(qo):
            np.minimum.at(qlo, qo, g.q_V)
            np.maximum.at(qhi, qo, g.q_V)
        emp = ~np.isfinite(qlo[:, 0])
        qlo[emp], qhi[emp] = g.o_c[emp, :2], g.o_c[emp, :2]
        g.o_pc = 0.5 * (qlo + qhi)
        g.o_prad = 0.5 * np.linalg.norm(qhi - qlo, axis=1) + 1.0
        # the alpha shadows' colour carries the alpha instead
        g.n_poly, g.n_obj = nP, nO
        g.o_kind = list(self.obj_kind)
        return g


class _Geo:
    """The flattened soup (see _Soup.finish) plus the trees (Props sets)."""


# --- local frames ---------------------------------------------------------
def _W(p, pts) -> np.ndarray:
    """(a along t, b along o, z) in a prop's frame -> world (k, 3)."""
    A = np.asarray(pts, dtype=np.float64)
    return np.column_stack([p.x + A[:, 0] * p.t[0] + A[:, 1] * p.o[0],
                            p.y + A[:, 0] * p.t[1] + A[:, 1] * p.o[1],
                            A[:, 2]])


def _dirs(p):
    t = np.array([p.t[0], p.t[1], 0.0])
    o = np.array([p.o[0], p.o[1], 0.0])
    return t, o


def _box(S, p, oid, a0, a1, b0, b1, z0, z1, rgb, faces='FBLRT', cols=None,
         plan=True, lo=0.0, hi=INF):
    """An axis-aligned box in the prop's frame. faces: F (b0, looks -o),
    B (b1, +o), L (a0, -t), R (a1, +t), T (top), D (bottom)."""
    t, o = _dirs(p)
    cols = cols or {}
    spec = {
        'F': ([(a0, b0, z0), (a1, b0, z0), (a1, b0, z1), (a0, b0, z1)], -o),
        'B': ([(a0, b1, z0), (a1, b1, z0), (a1, b1, z1), (a0, b1, z1)], o),
        'L': ([(a0, b0, z0), (a0, b1, z0), (a0, b1, z1), (a0, b0, z1)], -t),
        'R': ([(a1, b0, z0), (a1, b1, z0), (a1, b1, z1), (a1, b0, z1)], t),
        'T': ([(a0, b0, z1), (a1, b0, z1), (a1, b1, z1), (a0, b1, z1)], _Z),
        'D': ([(a0, b0, z0), (a1, b0, z0), (a1, b1, z0), (a0, b1, z0)], -_Z),
    }
    for f in faces:
        V, out = spec[f]
        S.face(_W(p, V), cols.get(f, rgb), oid, out, plan=plan, lo=lo, hi=hi)
    if plan and 'T' in faces and min(a1 - a0, b1 - b0) >= 4.0:
        _bevel(S, _W(p, spec['T'][0])[:, :2], cols.get('T', rgb), oid, z1)


def _bevel(S, poly2, rgb, oid, z):
    """Plan view: a flat roof reads as a flat grey card from above. A
    parapet stroke round it, LIT on the edges that face the sun and SHADED
    on the others, is what makes it a roof with a lit and a shaded side."""
    top = np.array(_shade(rgb, _Z), dtype=np.float64)
    c = poly2.mean(axis=0)
    for a2, b2 in zip(poly2, np.roll(poly2, -1, axis=0)):
        m = 0.5 * (a2 + b2)
        n = m - c
        n /= max(float(np.hypot(*n)), 1e-9)
        lit = float(n @ SUN_XY) > 0.0
        col = np.clip(top * (1.10 if lit else 0.70), 0, 255).astype(int)
        S.Q.append((oid, 1, np.vstack([a2, b2]), tuple(int(v) for v in col), 2, z + 0.01,
                    0.7, 1.0))


def _decal(S, p, oid, b, pts_az, rgb, side=-1.0, sub=1, lo=0.0, hi=INF):
    """A flat shape on a vertical face at depth b, (a, z) points, looking
    along side * o. Drawn after its face (sub)."""
    t, o = _dirs(p)
    bb = b + 0.03 * side
    V = _W(p, [(a, bb, z) for a, z in pts_az])
    S.face(V, rgb, oid, side * o, sub=sub, lo=lo, hi=hi, plan=False)


def _oct(cu, cv, r, n=8):
    return [(cu + r * math.cos(2 * math.pi * k / n + math.pi / n),
             cv + r * math.sin(2 * math.pi * k / n + math.pi / n)) for k in range(n)]


def _cast(S, lay, pts3, rmax, alpha=None):
    """The ground shadow of a set of world points (a solid's corners), as
    its own object (a shadow reaches further than its caster, so it culls on
    its own bounds). Opaque grass-in-shade beyond the run-off; alpha nearer."""
    hull = _shadow_of(pts3)
    if len(hull) < 3:
        return
    if alpha is None:
        alpha = float(lay.edge.exact(_outline(hull, 4.0), cap=RUNOFF_MAX + 4.0).min()) \
            < RUNOFF_MAX + 2.0
    S.ground(hull, S.obj(rmax), alpha=alpha)


def _rng_for(p) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f'{p.kind}{p.x:.2f}{p.y:.2f}'.encode()))


# --- builders, one per kind ----------------------------------------------
def _g_stand(S, p, lay):
    """Stepped seating (crowd speckle rows on the risers, a far proxy of one
    quad per row), a back wall, end walls, and -- roofed -- a cantilever roof
    on pillars with a coloured fascia. Chunked every ~12 m, so the painter
    orders a long stand against trees and posts correctly."""
    rng = _rng_for(p)
    roof = bool(p.extra.get('roof', True))
    livery = int(p.extra.get('livery', 0))
    fascia = ((38, 60, 110), (132, 44, 38), (62, 66, 74))[livery % 3]
    hl, hw = p.hl, p.hw
    bf, bb = -hw, hw
    z0 = 1.3
    rows = max(3, min(6, int(round((2 * hw - 0.6) / 1.6))))
    tread = (2 * hw - 0.6) / rows
    rise = 0.62 if roof else 0.55
    z_top = z0 + rows * rise
    zr_f, zr_b = z_top + 3.8, z_top + 2.9
    over = 2.6
    zb = zr_b if roof else z_top + 1.1
    nck = max(1, int(round(2 * hl / 12.0)))
    ch = 2 * hl / nck
    t, o = _dirs(p)
    cw = np.array(CROWD, dtype=np.float64)
    for k in range(nck):
        a0, a1 = -hl + k * ch, -hl + (k + 1) * ch
        oid = S.obj(R_BIG)
        S.face(_W(p, [(a0, bf, 0), (a1, bf, 0), (a1, bf, z0), (a0, bf, z0)]),
               C_CONCRETE, oid, -o)
        nb = max(2, int(round(ch / 3.0)))
        for i in range(rows):
            bi = bf + i * tread
            zl, zh = z0 + i * rise, z0 + (i + 1) * rise
            picks = rng.integers(0, len(CROWD), nb)
            for j in range(nb):
                aj0, aj1 = a0 + j * ch / nb, a0 + (j + 1) * ch / nb
                S.face(_W(p, [(aj0, bi, zl), (aj1, bi, zl), (aj1, bi, zh), (aj0, bi, zh)]),
                       CROWD[int(picks[j])], oid, -o, lo=0.0, hi=LOD_CROWD, plan=False)
            mean = tuple(int(v) for v in cw[picks].mean(axis=0))
            S.face(_W(p, [(a0, bi, zl), (a1, bi, zl), (a1, bi, zh), (a0, bi, zh)]),
                   mean, oid, -o, lo=LOD_CROWD, plan=False)
            # roofed: bare concrete treads (the roof hides them from above);
            # open bleachers: the crowd sits on them, and the plan view sees
            # them -- the row's own mean colour stripes the stand
            S.face(_W(p, [(a0, bi, zh), (a1, bi, zh), (a1, bi + tread, zh), (a0, bi + tread, zh)]),
                   C_TREAD if roof else mean, oid, _Z, plan=not roof)
        S.face(_W(p, [(a0, bb, 0), (a1, bb, 0), (a1, bb, zb), (a0, bb, zb)]),
               C_CONCRETE, oid, o)
        if roof:
            S.face(_W(p, [(a0, bb - 0.05, z_top), (a1, bb - 0.05, z_top),
                          (a1, bb - 0.05, zr_b), (a0, bb - 0.05, zr_b)]),
                   C_UNDER, oid, -o, plan=False)
            S.face(_W(p, [(a0, bf - over, zr_f), (a1, bf - over, zr_f),
                          (a1, bb + 0.3, zr_b), (a0, bb + 0.3, zr_b)]), C_ROOF, oid, _Z)
            S.face(_W(p, [(a0, bf - over, zr_f - 0.35), (a1, bf - over, zr_f - 0.35),
                          (a1, bb + 0.3, zr_b - 0.35), (a0, bb + 0.3, zr_b - 0.35)]),
                   C_UNDER, oid, -_Z, plan=False)
            S.face(_W(p, [(a0, bf - over, zr_f - 1.0), (a1, bf - over, zr_f - 1.0),
                          (a1, bf - over, zr_f), (a0, bf - over, zr_f)]), fascia, oid, -o,
                   plan=False)
            # plan: the fascia's colour along the roof's front edge, the
            # stand's identity from above
            e0 = _W(p, [(a0, bf - over + 0.4, 0.0)])[0, :2]
            e1 = _W(p, [(a1, bf - over + 0.4, 0.0)])[0, :2]
            S.pline(e0, e1, fascia, oid, 0.8, 2.0, 2, zr_f + 0.01)
            if k % 2 == 0 and ch > 6.0:
                _decal(S, p, oid, bf - over, [(a0 + 1.6, zr_f - 0.78), (a1 - 1.6, zr_f - 0.78),
                                              (a1 - 1.6, zr_f - 0.22), (a0 + 1.6, zr_f - 0.22)],
                       C_WHITE, hi=LOD_SPECKLE)
            for a in ([a0] + ([a1] if k == nck - 1 else [])):
                S.line(_W(p, [(a, bf + 0.35, z0)])[0], _W(p, [(a, bf + 0.35, zr_f - 0.35)])[0],
                       C_STEEL, oid, 0.32)
        prof = [(bf, 0.0), (bb, 0.0), (bb, zb), (bf, z0 + rise)]
        if k == 0:
            S.face(_W(p, [(a0, b, z) for b, z in prof]), C_CONCRETE_DK, oid, -t)
        if k == nck - 1:
            S.face(_W(p, [(a1, b, z) for b, z in prof]), C_CONCRETE_DK, oid, t)
    pts = [(a, b, z) for a in (-hl, hl) for b, z in ((bf, 0), (bb, 0), (bb, zb))]
    if roof:
        pts += [(a, bf - over, zr_f) for a in (-hl, hl)]
    _cast(S, lay, _W(p, pts), R_BIG)


def _g_pits(S, p, lay):
    """The pit building: garage doors on the track side, a glazed first floor
    (hospitality), a flat roof. 16 m chunks. `shed`: the proving ground's
    workshop row, the same bones in a lower, warmer build."""
    H = p.h
    if p.extra.get('shed'):
        return _g_shed(S, p, lay)
    hl, hw = p.hl, p.hw
    nck = max(1, int(round(2 * hl / 16.0)))
    ch = 2 * hl / nck
    for k in range(nck):
        a0, a1 = -hl + k * ch, -hl + (k + 1) * ch
        oid = S.obj(R_BIG)
        faces = 'FBT' + ('L' if k == 0 else '') + ('R' if k == nck - 1 else '')
        _box(S, p, oid, a0, a1, -hw, hw, 0.0, H, C_WHITE, faces=faces,
             cols={'T': C_ROOF, 'B': C_CONCRETE, 'L': C_CONCRETE, 'R': C_CONCRETE})
        am = 0.5 * (a0 + a1)
        dw = 0.33 * ch
        _decal(S, p, oid, -hw, [(am - dw, 0.0), (am + dw, 0.0), (am + dw, 4.3), (am - dw, 4.3)],
               C_DOOR)
        # a plain team-colour board over each garage (a colour, not a name)
        _decal(S, p, oid, -hw, [(am - 0.7 * dw, 4.45), (am + 0.7 * dw, 4.45),
                                (am + 0.7 * dw, 4.95), (am - 0.7 * dw, 4.95)],
               FLAGS[(k * 5 + 1) % len(FLAGS)], hi=LOD_SPECKLE)
        _decal(S, p, oid, -hw, [(a0 + 0.5, 5.1), (a1 - 0.5, 5.1), (a1 - 0.5, 6.9), (a0 + 0.5, 6.9)],
               C_GLASS)
        _decal(S, p, oid, -hw, [(a0, H - 0.55), (a1, H - 0.55), (a1, H), (a0, H)], C_ROOF_DK)
        for a in (am - dw * 0.34, am + dw * 0.34):
            _decal(S, p, oid, -hw, [(a - 0.06, 0.0), (a + 0.06, 0.0), (a + 0.06, 4.3),
                                    (a - 0.06, 4.3)], C_UNDER, sub=2, hi=LOD_SPECKLE)
        # rooftop plant: the roofline is not a ruler edge, and the plan view
        # gets something on the roof
        if k % 2 == 1:
            _box(S, p, oid, am - 2.2, am + 0.6, 0.5, 2.6, H, H + 1.3, C_STEEL, faces='FLRT',
                 hi=LOD_NEAR)
        else:
            _box(S, p, oid, am - 1.0, am + 1.0, -1.5, 0.2, H, H + 0.8, C_ROOF_DK, faces='FLRT',
                 hi=LOD_NEAR)
    _cast(S, lay, _W(p, [(a, b, z) for a in (-hl, hl) for b in (-hw, hw) for z in (0, H)]),
          R_BIG)


def _g_shed(S, p, lay):
    """A workshop row: roller doors, a window band, a shallow roof."""
    H = p.h
    hl, hw = p.hl, p.hw
    nck = max(1, int(round(2 * hl / 12.0)))
    ch = 2 * hl / nck
    for k in range(nck):
        a0, a1 = -hl + k * ch, -hl + (k + 1) * ch
        oid = S.obj(R_BIG)
        faces = 'FBT' + ('L' if k == 0 else '') + ('R' if k == nck - 1 else '')
        _box(S, p, oid, a0, a1, -hw, hw, 0.0, H, C_OFFICE, faces=faces,
             cols={'T': C_ROOF_DK})
        am = 0.5 * (a0 + a1)
        _decal(S, p, oid, -hw, [(am - 3.4, 0.0), (am + 0.6, 0.0), (am + 0.6, 4.0), (am - 3.4, 4.0)],
               (150, 154, 158))
        _decal(S, p, oid, -hw, [(am + 1.4, 1.2), (a1 - 0.8, 1.2), (a1 - 0.8, 2.6), (am + 1.4, 2.6)],
               C_GLASS)
        _decal(S, p, oid, -hw, [(a0, H - 0.5), (a1, H - 0.5), (a1, H), (a0, H)], (70, 110, 150))
    _cast(S, lay, _W(p, [(a, b, z) for a in (-hl, hl) for b in (-hw, hw) for z in (0, H)]),
          R_BIG)


CAR_TONES = ((226, 226, 222), (170, 174, 178), (46, 62, 96), (66, 68, 74), (128, 50, 44),
             (196, 190, 176), (40, 84, 70))


def _g_car(S, p, lay):
    """A parked car: a generic hatchback volume (body, glasshouse, dark
    wheel band). No marque, no badge."""
    oid = S.obj(R_SMALL, detail=True)
    col = CAR_TONES[int(p.extra.get('tone', 0)) % len(CAR_TONES)]
    _box(S, p, oid, -2.05, 2.05, -0.84, 0.84, 0.3, 0.92, col, faces='FBLRT')
    _box(S, p, oid, -1.25, 0.75, -0.74, 0.74, 0.92, 1.42, (58, 68, 84), faces='FBLR')
    _box(S, p, oid, -1.3, 0.8, -0.76, 0.76, 1.40, 1.46, col, faces='T')
    _box(S, p, oid, -1.95, 1.95, -0.8, 0.8, 0.0, 0.3, C_TYRE, faces='FBLR', plan=False)
    _cast(S, lay, _W(p, [(a, b, z) for a in (-2.05, 2.05) for b in (-0.84, 0.84)
                         for z in (0.0, 1.2)]), R_SMALL)


def _g_tower(S, p, lay):
    """Race control / the airfield tower: a two-storey glazed base (the
    circuit's), a slim shaft with window slits, a cantilevered glass cab with
    mullions, a flat roof with an overhang, an aerial. The slim variant (the
    proving ground's) has no base."""
    w = p.hl
    H = p.h
    slim = bool(p.extra.get('slim', False))
    oid = S.obj(R_BIG)
    zc = H - 4.2
    ws = (0.55 if slim else 0.5) * w
    if not slim:
        _box(S, p, oid, -w, w, -w, w, 0.0, 6.4, C_WHITE, faces='FBLRT', cols={'T': C_ROOF})
        for sgn in (-1.0, 1.0):
            _decal(S, p, oid, sgn * w, [(-w + 0.5, 3.6), (w - 0.5, 3.6), (w - 0.5, 5.6),
                                        (-w + 0.5, 5.6)], C_GLASS, side=sgn)
            _decal(S, p, oid, sgn * w, [(-w + 0.5, 0.6), (w - 0.5, 0.6), (w - 0.5, 2.4),
                                        (-w + 0.5, 2.4)], C_GLASS, side=sgn)
        z_sh = 6.4
    else:
        z_sh = 0.0
    _box(S, p, oid, -ws, ws, -ws, ws, z_sh, zc, C_CONCRETE, faces='FBLR')
    for sgn in (-1.0, 1.0):
        for z in np.arange(z_sh + 2.2, zc - 1.0, 3.2):
            _decal(S, p, oid, sgn * ws, [(-0.25, z), (0.25, z), (0.25, z + 1.4), (-0.25, z + 1.4)],
                   C_GLASS, side=sgn, hi=LOD_SPECKLE)
    wc = w + 0.6
    _box(S, p, oid, -wc, wc, -wc, wc, zc - 0.5, zc, C_WHITE, faces='FBLRD')
    _box(S, p, oid, -wc, wc, -wc, wc, zc, zc + 3.0, C_GLASS, faces='FBLR')
    for sgn in (-1.0, 1.0):
        for a in np.linspace(-wc, wc, 6)[1:-1]:
            _decal(S, p, oid, sgn * wc, [(a - 0.08, zc), (a + 0.08, zc), (a + 0.08, zc + 3.0),
                                         (a - 0.08, zc + 3.0)], C_STEEL_DK, side=sgn,
                   hi=LOD_SPECKLE)
    wr = w + 1.3
    _box(S, p, oid, -wr, wr, -wr, wr, zc + 3.0, zc + 3.6, C_ROOF_DK, faces='FBLRTD',
         cols={'T': C_ROOF})
    S.line(_W(p, [(0.3 * w, 0.2 * w, zc + 3.6)])[0], _W(p, [(0.3 * w, 0.2 * w, H + 4.0)])[0],
           C_STEEL, oid, 0.12)
    _cast(S, lay, _W(p, [(a, b, z) for a in (-wr, wr) for b in (-wr, wr) for z in (0.0, zc + 3.6)]),
          R_BIG)


def _g_trailer(S, p, lay):
    """A paddock transporter: a white box trailer with a coloured stripe and
    a cab in the team colour. Generic: no livery, no marque."""
    tone = FLAGS[int(p.extra.get('tone', 0)) % len(FLAGS)]
    hl, hw = p.hl, p.hw
    oid = S.obj(R_SMALL, detail=True)
    _box(S, p, oid, -hl, hl - 2.7, -hw, hw, 0.35, p.h, C_WHITE, faces='FBLRT',
         cols={'T': C_ROOF})
    for sgn in (-1.0, 1.0):
        _decal(S, p, oid, sgn * hw, [(-hl + 0.4, 1.5), (hl - 3.1, 1.5), (hl - 3.1, 2.1),
                                     (-hl + 0.4, 2.1)], tone, side=sgn)
    _box(S, p, oid, hl - 2.5, hl, -hw, hw, 0.35, 3.1, tone, faces='FBRT')
    _box(S, p, oid, -hl, hl, -hw + 0.1, hw - 0.1, 0.0, 0.35, C_TYRE, faces='FB', plan=False)
    _cast(S, lay, _W(p, [(a, b, z) for a in (-hl, hl) for b in (-hw, hw) for z in (0, p.h)]),
          R_SMALL)


def _g_flood(S, p, lay):
    """A floodlight tower: a pole and a lamp head facing the track."""
    oid = S.obj(R_BIG)
    H = p.h
    S.line(_W(p, [(0, 0, 0)])[0], _W(p, [(0, 0, H - 1.6)])[0], C_POLE, oid, 0.55)
    _box(S, p, oid, -1.7, 1.7, -0.3, 0.3, H - 1.9, H, C_STEEL_DK, faces='FBLRT')
    for r_ in range(2):
        for c_ in range(4):
            a = -1.3 + c_ * 0.86
            z = H - 1.65 + r_ * 0.8
            _decal(S, p, oid, -0.3, [(a - 0.28, z), (a + 0.28, z), (a + 0.28, z + 0.55),
                                     (a - 0.28, z + 0.55)], (236, 234, 222), hi=LOD_POSTS)
    _decal(S, p, oid, -0.3, [(-1.6, H - 1.7), (1.6, H - 1.7), (1.6, H - 0.2), (-1.6, H - 0.2)],
           (196, 196, 188), lo=LOD_POSTS)
    _cast(S, lay, _W(p, [(a, b, z) for a in (-1.7, 1.7) for b in (-0.3, 0.3)
                         for z in (H - 1.9, H)]), R_BIG)


def _g_lightpole(S, p, lay):
    oid = S.obj(R_BIG)
    H = p.h
    S.line(_W(p, [(0, 0, 0)])[0], _W(p, [(0, 0, H)])[0], C_POLE, oid, 0.3)
    _box(S, p, oid, -0.7, 0.7, -0.6, 0.1, H - 0.45, H, C_STEEL_DK, faces='FBLRT')
    _decal(S, p, oid, -0.6, [(-0.55, H - 0.4), (0.55, H - 0.4), (0.55, H - 0.08),
                             (-0.55, H - 0.08)], (236, 234, 222), hi=LOD_NEAR)


def _g_flag(S, p, lay):
    oid = S.obj(R_SMALL, detail=True)
    H = p.h
    tone = FLAGS[int(p.extra.get('tone', 0)) % len(FLAGS)]
    t, o = _dirs(p)
    S.line(_W(p, [(0, 0, 0)])[0], _W(p, [(0, 0, H)])[0], C_WHITE, oid, 0.12)
    V = _W(p, [(0.05, 0, H - 1.25), (1.9, 0, H - 1.15), (1.9, 0, H - 0.1), (0.05, 0, H - 0.05)])
    S.face(V, tone, oid, -o, two=True, plan=False)
    S.face(V, tone, oid, o, two=False, plan=False)


def _g_board(S, p, lay):
    """A FICTIONAL sponsor board behind the barrier: abstract colour blocks."""
    bg, shapes = DESIGNS[int(p.extra.get('design', 0)) % len(DESIGNS)]
    oid = S.obj(R_MED)
    hl, hw = p.hl, p.hw
    z0, z1 = 1.15, 2.45
    for a in (-hl + 0.5, hl - 0.5):
        S.line(_W(p, [(a, 0.0, 0.0)])[0], _W(p, [(a, 0.0, z0)])[0], C_STEEL_DK, oid, 0.1,
               hi=LOD_POSTS)
    _box(S, p, oid, -hl, hl, -hw, hw, z0, z1, bg, faces='FB', cols={'B': C_BOARD_BACK})
    _box(S, p, oid, -hl, hl, -hw, hw, z0, z1, C_BOARD_BACK, faces='T', hi=LOD_TOP)
    W_, H_ = 2 * hl, z1 - z0

    def az(u, v):
        return (-hl + u * W_, z0 + v * H_)
    for shape, col in shapes:
        polys = []
        if shape == 'bar':
            polys = [[az(0.40, 0.30), az(0.94, 0.30), az(0.94, 0.70), az(0.40, 0.70)]]
        elif shape == 'disc':
            polys = [[(-hl + 0.18 * W_ + x * 1.0, z0 + 0.5 * H_ + y * 1.0)
                      for x, y in _oct(0.0, 0.0, 0.36 * H_)]]
        elif shape == 'chev':
            for c in (0.25, 0.47, 0.69):
                polys.append([az(c - 0.08, 0.15), az(c, 0.15), az(c + 0.07, 0.5), az(c, 0.85),
                              az(c - 0.08, 0.85), az(c - 0.01, 0.5)])
        elif shape == 'split':
            polys = [[az(0.58, 0.0), az(1.0, 0.0), az(1.0, 1.0), az(0.80, 1.0)]]
        elif shape == 'band':
            polys = [[az(0.05, 0.58), az(0.95, 0.58), az(0.95, 0.78), az(0.05, 0.78)]]
        elif shape == 'sq':
            polys = [[az(0.06, 0.18), az(0.16, 0.18), az(0.16, 0.48), az(0.06, 0.48)]]
        elif shape == 'dots':
            for c, cc in ((0.25, (40, 40, 48)), (0.5, (200, 90, 40)), (0.75, (60, 110, 170))):
                pts = [(-hl + c * W_ + x, z0 + 0.5 * H_ + y) for x, y in _oct(0.0, 0.0, 0.3 * H_)]
                _decal(S, p, oid, -hw, pts, cc, hi=LOD_SPECKLE)
            continue
        elif shape == 'zig':
            polys = [[az(0.05, 0.35), az(0.3, 0.65), az(0.55, 0.35), az(0.8, 0.65), az(0.95, 0.5),
                      az(0.95, 0.68), az(0.8, 0.83), az(0.55, 0.53), az(0.3, 0.83), az(0.05, 0.53)]]
        elif shape == 'stripes':
            for c in (0.6, 0.72, 0.84):
                polys.append([az(c, 0.1), az(c + 0.06, 0.1), az(c + 0.06, 0.9), az(c, 0.9)])
        for pts in polys:
            _decal(S, p, oid, -hw, pts, col, hi=LOD_SPECKLE)
    S.pline(_W(p, [(-hl, 0, 0)])[0, :2], _W(p, [(hl, 0, 0)])[0, :2], bg, oid, 0.45, 2.0, 1, 2.4)


def _g_brake(S, p, lay):
    """A 300 / 200 / 100 m board: white, 3 / 2 / 1 black diagonal stripes,
    on a post, facing the braking car."""
    oid = S.obj(R_MED)
    n = int(p.extra.get('stripes', 1))
    S.line(_W(p, [(0, 0, 0)])[0], _W(p, [(0, 0, 1.3)])[0], C_STEEL_DK, oid, 0.14)
    _box(S, p, oid, -0.8, 0.8, -0.06, 0.06, 1.3, 2.9, C_WHITE, faces='FBT',
         cols={'B': C_BOARD_BACK, 'T': C_BOARD_BACK})
    for k in range(n):
        u = (k + 1) / (n + 1) * 1.6 - 0.8
        _decal(S, p, oid, -0.06, [(u - 0.26 - 0.1, 1.42), (u - 0.26 + 0.1, 1.42),
                                  (u + 0.26 + 0.1, 2.78), (u + 0.26 - 0.1, 2.78)], (30, 30, 34))
    S.pline(_W(p, [(-0.8, 0, 0)])[0, :2], _W(p, [(0.8, 0, 0)])[0, :2], C_WHITE, oid, 0.3, 2.0, 1, 2.9)


def _g_marshal(S, p, lay):
    """A marshal post: a white booth with an orange band, a flat roof, and
    a pole with a plain orange flag."""
    oid = S.obj(R_SMALL)
    hl, hw = p.hl, p.hw
    _box(S, p, oid, -hl, hl, -hw, hw, 0.0, 2.3, C_WHITE, faces='FBLR')
    _decal(S, p, oid, -hw, [(-hl, 1.7), (hl, 1.7), (hl, 2.1), (-hl, 2.1)], (214, 120, 42))
    _decal(S, p, oid, -hw, [(-0.6 * hl, 0.9), (0.6 * hl, 0.9), (0.6 * hl, 1.55), (-0.6 * hl, 1.55)],
           C_GLASS)
    _box(S, p, oid, -hl - 0.3, hl + 0.3, -hw - 0.3, hw + 0.3, 2.3, 2.55, C_ROOF_DK,
         faces='FBLRT', cols={'T': C_ROOF})
    fid = S.obj(R_SMALL, detail=True)
    a, b = hl - 0.2, hw + 0.6
    S.line(_W(p, [(a, b, 0)])[0], _W(p, [(a, b, 5.2)])[0], C_WHITE, fid, 0.09)
    t, o = _dirs(p)
    V = _W(p, [(a - 0.05, b, 4.45), (a - 1.25, b, 4.5), (a - 1.25, b, 5.15), (a - 0.05, b, 5.2)])
    S.face(V, (230, 120, 40), fid, -o, two=True, plan=False)
    _cast(S, lay, _W(p, [(a_, b_, z) for a_ in (-hl - 0.3, hl + 0.3) for b_ in (-hw - 0.3, hw + 0.3)
                         for z in (0, 2.55)]), R_SMALL)


def _g_hangar(S, p, lay):
    """A barrel-roofed hangar: the arch spans the door wall (along t), the
    barrel runs back along o; a big panelled door faces the pad."""
    oid = S.obj(R_BIG)
    hl, hw, H = p.hl, p.hw, p.h
    na = 9
    th = np.linspace(0.0, math.pi, na + 1)
    arch = [(-hl * math.cos(a), H * math.sin(a) ** 0.85) for a in th]
    t, o = _dirs(p)
    tone = (C_HANGAR, (170, 176, 170), (190, 184, 170))[int(p.extra.get('tone', 0)) % 3]
    for k in range(na):
        (a0, z0), (a1, z1) = arch[k], arch[k + 1]
        V = _W(p, [(a0, -hw, z0), (a1, -hw, z1), (a1, hw, z1), (a0, hw, z0)])
        mid = np.array([0.5 * (a0 + a1), 0.0, 0.5 * (z0 + z1)])
        out = mid[0] * t + mid[2] * _Z
        S.face(V, tone, oid, out)
    S.face(_W(p, [(a, -hw, z) for a, z in arch]), C_WHITE, oid, -o)
    S.face(_W(p, [(a, hw, z) for a, z in arch]), C_WHITE, oid, o)
    dh = 0.62 * H
    dw = 0.64 * hl
    _decal(S, p, oid, -hw, [(-dw, 0.0), (dw, 0.0), (dw, dh), (-dw, dh)], C_HANGAR_DOOR)
    for a in np.linspace(-dw, dw, 7)[1:-1]:
        _decal(S, p, oid, -hw, [(a - 0.07, 0.0), (a + 0.07, 0.0), (a + 0.07, dh), (a - 0.07, dh)],
               C_UNDER, sub=2, hi=LOD_NEAR)
    _decal(S, p, oid, -hw, [(-dw - 0.4, dh), (dw + 0.4, dh), (dw + 0.4, dh + 0.7),
                            (-dw - 0.4, dh + 0.7)], C_ROOF_DK, sub=1)
    _cast(S, lay, _W(p, [(a, b, z) for a, z in arch for b in (-hw, hw)]), R_BIG)


def _g_office(S, p, lay):
    """A pitched-roof building (the test centre's office, the skidpad's
    timing hut): walls, window bands, a red-tiled roof, gable ends."""
    oid = S.obj(R_BIG)
    hl, hw, H = p.hl, p.hw, p.h
    hut = bool(p.extra.get('hut', False))
    t, o = _dirs(p)
    _box(S, p, oid, -hl, hl, -hw, hw, 0.0, H, C_OFFICE, faces='FB')
    zr = H + (1.6 if hut else 2.6)
    for a, sgn in ((-hl, -1.0), (hl, 1.0)):
        S.face(_W(p, [(a, -hw, 0), (a, hw, 0), (a, hw, H), (a, 0, zr), (a, -hw, H)]),
               C_OFFICE, oid, sgn * t)
    ov = 0.5
    S.face(_W(p, [(-hl - ov, -hw - ov, H - 0.35), (hl + ov, -hw - ov, H - 0.35),
                  (hl + ov, 0, zr), (-hl - ov, 0, zr)]), C_ROOF_RED, oid, -o + _Z)
    S.face(_W(p, [(-hl - ov, hw + ov, H - 0.35), (hl + ov, hw + ov, H - 0.35),
                  (hl + ov, 0, zr), (-hl - ov, 0, zr)]), C_ROOF_RED, oid, o + _Z)
    bands = ((1.0, 2.2),) if (hut or H < 6.0) else ((1.0, 2.2), (4.0, 5.2))
    for sgn in (-1.0, 1.0):
        for z0, z1 in bands:
            _decal(S, p, oid, sgn * hw, [(-hl + 1.0, z0), (hl - 1.0, z0), (hl - 1.0, z1),
                                         (-hl + 1.0, z1)], C_GLASS, side=sgn)
        if sgn < 0:
            _decal(S, p, oid, -hw, [(-0.7, 0.0), (0.7, 0.0), (0.7, 2.2), (-0.7, 2.2)], C_DOOR,
                   sub=2)
    _cast(S, lay, _W(p, [(a, b, z) for a in (-hl - ov, hl + ov) for b in (-hw - ov, hw + ov)
                         for z in (0, H)] + [(a, 0, zr) for a in (-hl, hl)]), R_BIG)


def _g_windsock(S, p, lay):
    oid = S.obj(R_SMALL, detail=True)
    H = p.h
    S.line(_W(p, [(0, 0, 0)])[0], _W(p, [(0, 0, H)])[0], C_WHITE, oid, 0.12)
    d = np.array([math.cos(math.radians(205.0)), math.sin(math.radians(205.0)), -0.12])
    d /= np.linalg.norm(d)
    side = np.cross(d, _Z)
    side /= np.linalg.norm(side)
    up = np.cross(side, d)
    top = np.array([p.x, p.y, H - 0.1])
    for k, (r0, r1, col) in enumerate(((0.45, 0.36, C_CONE), (0.36, 0.28, C_WHITE),
                                       (0.28, 0.2, C_CONE))):
        c0, c1 = top + d * (1.1 * k), top + d * (1.1 * (k + 1))
        for j in range(4):
            a0, a1 = 0.5 * math.pi * j, 0.5 * math.pi * (j + 1)
            e0 = math.cos(a0) * side + math.sin(a0) * up
            e1 = math.cos(a1) * side + math.sin(a1) * up
            V = np.array([c0 + r0 * e0, c1 + r1 * e0, c1 + r1 * e1, c0 + r0 * e1])
            S.face(V, col, oid, 0.5 * (e0 + e1), plan=False, two=True)


def _g_cones(S, p, lay):
    """Parked stacks of cones: each a slim orange frustum."""
    oid = S.obj(R_SMALL, detail=True)
    rng = _rng_for(p)
    n = int(p.extra.get('n', 5))
    for k in range(n):
        a = -p.hl + 0.4 + (2 * p.hl - 0.8) * rng.random()
        b = -p.hw + 0.3 + (2 * p.hw - 0.6) * rng.random()
        h = 0.55 + 0.4 * rng.random()
        r0, r1 = 0.22, 0.07
        c = _W(p, [(a, b, 0.0)])[0]
        for j in range(4):
            q0, q1 = 0.5 * math.pi * j + 0.4, 0.5 * math.pi * (j + 1) + 0.4
            e0 = np.array([math.cos(q0), math.sin(q0), 0.0])
            e1 = np.array([math.cos(q1), math.sin(q1), 0.0])
            V = np.array([c + r0 * e0, c + r0 * e1, c + r1 * e1 + h * _Z, c + r1 * e0 + h * _Z])
            S.face(V, C_CONE, oid, e0 + e1 + 0.3 * _Z, plan=False)
        S.Q.append((oid, 2, c[None, :2].copy(), C_CONE, 3, h, 0.22, 1.5))


def _g_pillar(S, p, lay):
    oid = S.obj(R_BIG)
    _box(S, p, oid, -p.hl, p.hl, -p.hw, p.hw, 0.0, p.h, C_STEEL, faces='FBLRT')


def _g_gantry(S, p, lay):
    """The start / finish gantry's beam (the pillars are their own props):
    a box across the road at 6.3-7.3 m in three pieces (so the near-plane
    drop takes a third at a time), five light pods over the road facing the
    grid -- lights OUT, dark red -- and a shadow across the tarmac."""
    A = np.array(p.extra['a'])
    B = np.array(p.extra['b'])
    road = np.array(p.extra['road'])
    span = float(np.hypot(*(B - A)))
    u = (B - A) / span
    f3 = np.array([road[0], road[1], 0.0])
    u3 = np.array([u[0], u[1], 0.0])
    z0, z1 = 6.3, 7.3
    dep = 0.5
    base = np.array([A[0], A[1], 0.0])
    for k in range(3):
        oid = S.obj(R_BIG)
        s0, s1 = span * k / 3.0, span * (k + 1) / 3.0

        def P(s, d, z):
            return base + s * u3 + d * f3 + z * _Z
        S.face([P(s0, -dep, z0), P(s1, -dep, z0), P(s1, -dep, z1), P(s0, -dep, z1)],
               (40, 52, 88), oid, -f3)
        S.face([P(s0, dep, z0), P(s1, dep, z0), P(s1, dep, z1), P(s0, dep, z1)],
               (40, 52, 88), oid, f3)
        S.face([P(s0, -dep, z1), P(s1, -dep, z1), P(s1, dep, z1), P(s0, dep, z1)],
               C_STEEL, oid, _Z)
        S.face([P(s0, -dep, z0), P(s1, -dep, z0), P(s1, dep, z0), P(s0, dep, z0)],
               C_UNDER, oid, -_Z, plan=False)
        for d, sgn in ((-dep, -1.0), (dep, 1.0)):
            a0, a1 = s0 + 0.4, s1 - 0.4
            V = [P(a0, d + 0.03 * sgn, z0 + 0.62), P(a1, d + 0.03 * sgn, z0 + 0.62),
                 P(a1, d + 0.03 * sgn, z0 + 0.86), P(a0, d + 0.03 * sgn, z0 + 0.86)]
            S.face(V, C_WHITE, oid, sgn * f3, sub=1, plan=False, hi=LOD_SPECKLE)
        if k == 1:
            for j in range(5):
                c = 0.5 * span + (j - 2) * 1.05
                q0, q1 = c - 0.26, c + 0.26
                for d0, d1, zz0, zz1, out in ((-0.2, -0.2, 5.15, 6.3, -f3), (0.2, 0.2, 5.15, 6.3, f3)):
                    S.face([P(q0, d0, zz0), P(q1, d0, zz0), P(q1, d0, zz1), P(q0, d0, zz1)],
                           C_HOUSING, oid, out, plan=False)
                for zc in (5.45, 5.95):
                    V = [P(c + x, -0.23, zc + y) for x, y in _oct(0.0, 0.0, 0.14)]
                    S.face(V, C_LAMP_OFF, oid, -f3, sub=1, plan=False)
    pts = [base + s * u3 + d * f3 + z * _Z for s in (0.0, span) for d in (-dep, dep)
           for z in (z0, z1)]
    _cast(S, lay, np.array(pts), R_BIG, alpha=True)


def _g_line_chunk(p):
    """World corners of a barrier chunk: front line P0 P1, back line Q0 Q1."""
    p0, p1 = np.array(p.extra['p0']), np.array(p.extra['p1'])
    o = np.array(p.extra['out'])
    T = 2.0 * p.hw
    return p0, p1, p0 + o * T, p1 + o * T, np.array([o[0], o[1], 0.0])


def _g_tyres(S, p, lay):
    """A tyre wall: black stacks, a coloured conveyor-belt cover on the upper
    front (alternating per chunk), a dark top."""
    oid = S.obj(R_MED)
    p0, p1, q0, q1, o3 = _g_line_chunk(p)
    H = p.h
    zb = 0.6 * H
    belt = C_BELTS[(int(p.extra.get('belt', 0)) + 2 * (int(abs(p.x)) // 97 % 2)) % 4]

    def V(a, b, z0, z1):
        return [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
    S.face(V(p0, p1, 0.0, zb), C_TYRE, oid, -o3, hi=LOD_NEAR)
    S.face(V(p0, p1, zb, H), belt, oid, -o3, hi=LOD_NEAR)
    far = tuple(int(0.55 * a + 0.45 * b) for a, b in zip(C_TYRE, belt))
    S.face(V(p0, p1, 0.0, H), far, oid, -o3, lo=LOD_NEAR, plan=False)
    S.face([(p0[0], p0[1], H), (p1[0], p1[1], H), (q1[0], q1[1], H), (q0[0], q0[1], H)],
           C_TYRE_TOP, oid, _Z, hi=LOD_TOP)
    S.face(V(q0, q1, 0.0, H), C_TYRE, oid, o3)
    # plan: the belt colour over the front half of the dark top, or the
    # wall reads as a second, narrower road
    fo = o3[:2] * 0.35
    S.pline(p0 + fo, p1 + fo, belt, oid, 0.7, 2.0, 2, H + 0.01)


def _g_armco(S, p, lay):
    """Armco: a galvanised rail on posts."""
    oid = S.obj(R_MED)
    p0, p1, q0, q1, o3 = _g_line_chunk(p)

    def V(a, b, z0, z1):
        return [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
    S.face(V(p0, p1, 0.42, 0.80), C_STEEL, oid, -o3, plan=False)
    m0, m1 = p0 + 0.05 * o3[:2], p1 + 0.05 * o3[:2]
    S.face(V(m0, m1, 0.42, 0.80), C_STEEL_DK, oid, o3, plan=False)
    S.line((p0[0] + 0.15 * o3[0], p0[1] + 0.15 * o3[1], 0.0),
           (p0[0] + 0.15 * o3[0], p0[1] + 0.15 * o3[1], 0.78), C_STEEL_DK, oid, 0.12,
           hi=LOD_POSTS, plan=False)
    S.pline(p0, p1, C_STEEL, oid, 0.35, 2.0, 1, 0.8)


def _g_wall(S, p, lay):
    """The dragstrip's low concrete wall, with its shadow (alpha: it stands
    on the verge, inside the run-off band)."""
    oid = S.obj(R_MED)
    p0, p1, q0, q1, o3 = _g_line_chunk(p)
    H = p.h

    def V(a, b, z0, z1):
        return [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
    S.face(V(p0, p1, 0.0, H), C_CONCRETE, oid, -o3)
    S.face([(p0[0], p0[1], H), (p1[0], p1[1], H), (q1[0], q1[1], H), (q0[0], q0[1], H)],
           C_CONCRETE, oid, _Z)
    S.face(V(q0, q1, 0.0, H), C_CONCRETE, oid, o3)
    _decal(S, Prop('d', 'wall', 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]),
                   (p1 - p0) / max(np.hypot(*(p1 - p0)), 1e-9), 0, 0, 0),
           oid, 0.0, [(-0.5 * np.hypot(*(p1 - p0)), 0.0), (0.5 * np.hypot(*(p1 - p0)), 0.0),
                      (0.5 * np.hypot(*(p1 - p0)), 0.12), (-0.5 * np.hypot(*(p1 - p0)), 0.12)],
           (120, 118, 112), side=-1.0 if _cross2(p1 - p0, o3[:2]) > 0 else 1.0,
           hi=LOD_NEAR)
    S.pline(p0, p1, C_CONCRETE, oid, 0.6, 2.0, 1, H)
    _cast(S, lay, np.array([(x, y, z) for x, y in (p0, p1, q0, q1) for z in (0.0, H)]),
          R_MED, alpha=True)


def _g_fence(S, p, lay):
    """Chain-link / catch fence: posts every ~4 m (near only) and two wires."""
    oid = S.obj(R_MED)
    p0, p1, q0, q1, o3 = _g_line_chunk(p)
    H = p.h
    L = float(np.hypot(*(p1 - p0)))
    n = max(1, int(round(L / 4.0)))
    col = (120, 124, 128)
    for k in range(n):
        q = p0 + (p1 - p0) * (k / n)
        S.line((q[0], q[1], 0.0), (q[0], q[1], H), col, oid, 0.09, hi=LOD_POSTS, plan=False)
    for z in (H, 0.55 * H):
        S.line((p0[0], p0[1], z), (p1[0], p1[1], z), col, oid, 0.05, plan=False)
    S.pline(p0, p1, (116, 120, 124), oid, 0.12, 1.0, 1, H)


def _g_xmas(S, p, lay):
    """The drag start-light 'tree': two columns of staging (pale), three
    amber, a green and a red lamp, on a post, facing the launch line. All
    lamps dark-ish: nothing is lit in a frame that does not know the start
    sequence."""
    oid = S.obj(R_MED)
    S.line(_W(p, [(0, 0, 0)])[0], _W(p, [(0, 0, 2.1)])[0], C_STEEL_DK, oid, 0.16)
    _box(S, p, oid, -0.35, 0.35, -0.12, 0.12, 2.0, 4.0, C_HOUSING, faces='FBLRT')
    lamps = ((3.78, (214, 208, 186)), (3.56, (214, 208, 186)), (3.28, C_AMBER), (3.02, C_AMBER),
             (2.76, C_AMBER), (2.48, C_GO), (2.2, C_LAMP_RED))
    for a in (-0.16, 0.16):
        for z, col in lamps:
            _decal(S, p, oid, -0.12, _oct(a, z, 0.085), col, hi=LOD_NEAR)


def _g_timing(S, p, lay):
    """A timing board at a gate: a dark scoreboard on two posts with amber
    segment blocks (no digits: the renderer does not know the time)."""
    oid = S.obj(R_MED)
    hl, hw = p.hl, p.hw
    for a in (-hl + 0.4, hl - 0.4):
        S.line(_W(p, [(a, 0, 0)])[0], _W(p, [(a, 0, 3.4)])[0], C_STEEL_DK, oid, 0.16)
    _box(S, p, oid, -hl, hl, -hw, hw, 3.4, 5.4, C_HOUSING, faces='FBT',
         cols={'B': C_BOARD_BACK})
    for j in range(5):
        a = -hl + 0.5 + j * (2 * hl - 1.0) / 5.0
        w = (2 * hl - 1.0) / 5.0 * 0.7
        _decal(S, p, oid, -hw, [(a, 3.75), (a + w, 3.75), (a + w, 3.9), (a, 3.9)], C_AMBER,
               hi=LOD_NEAR)
        _decal(S, p, oid, -hw, [(a, 4.9), (a + w, 4.9), (a + w, 5.05), (a, 5.05)], C_AMBER,
               hi=LOD_NEAR)
        _decal(S, p, oid, -hw, [(a, 3.9), (a + 0.15, 3.9), (a + 0.15, 4.9), (a, 4.9)], C_AMBER,
               hi=LOD_NEAR)


def _g_gatepost(S, p, lay):
    oid = S.obj(R_MED)
    S.line(_W(p, [(0, 0, 0)])[0], _W(p, [(0, 0, p.h)])[0], C_WHITE, oid, 0.16)
    _box(S, p, oid, -0.15, 0.15, -0.15, 0.15, 0.45, 0.85, C_HOUSING, faces='FBLRT')
    _box(S, p, oid, -0.12, 0.12, -0.12, 0.12, p.h - 0.6, p.h, (214, 120, 42), faces='FBLRT')


BUILDERS = {'stand': _g_stand, 'pits': _g_pits, 'tower': _g_tower, 'trailer': _g_trailer,
            'flood': _g_flood, 'lightpole': _g_lightpole, 'flag': _g_flag, 'board': _g_board,
            'brake': _g_brake, 'marshal': _g_marshal, 'hangar': _g_hangar,
            'office': _g_office, 'windsock': _g_windsock, 'cones': _g_cones,
            'pillar': _g_pillar, 'gantry': _g_gantry, 'tyres': _g_tyres, 'armco': _g_armco,
            'wall': _g_wall, 'fence': _g_fence, 'xmas': _g_xmas, 'timing': _g_timing,
            'gatepost': _g_gatepost, 'car': _g_car}


def _build_geo(lay) -> _Geo:
    S = _Soup()
    for p in lay.props:
        S.kind = p.kind
        BUILDERS[p.kind](S, p, lay)
    g = S.finish()
    return g


# ======================================================================= #
#  TREE SPRITES                                                           #
# ======================================================================= #
_LIGHT_VEC = {'F': (0.0, 0.50, 0.78),     # sun behind the viewer, high
              'L': (-0.82, 0.42, 0.40),   # sun on the viewer's left
              'K': (0.0, 0.50, -0.87)}    # looking into the sun: backlit


def _mix(c0, c1, f):
    return tuple(int(round(a + (b - a) * f)) for a, b in zip(c0, c1))


def _canopy(surf, rng, cx, cy, rx, ry, n, rmin, rmax, cols, L, dim=1.0, warm=True,
            leaf=(0.07, 0.14), flecks=1.0):
    """A crown: a SILHOUETTE of n large blobs in the shade tone (it gives the
    outline its lumps), then ~3n small LEAF blobs shaded by a sphere normal
    at their position against the light L (image x right, y up, z toward the
    viewer), back ones first, then a few flecks on the lit side. Every blob
    is sampled INSIDE the crown ellipse less its own radius, so nothing
    floats off the outline and nothing is cut by the image edge. Every
    random draw is made BEFORE the light is applied, so the F / L / K
    variants of one shape have the identical outline and only the light
    moves (a tree that changes variant as the camera turns does not jump)."""
    dark, mid, light = cols
    if warm:     # late-afternoon sun: the lit leaves go a touch golden
        light = (min(255, light[0] + 6), min(255, light[1] + 3), max(0, light[2] - 4))
    Lv = np.array(L, dtype=np.float64)
    Lv /= np.linalg.norm(Lv)
    m = min(rx, ry)

    def sample(k, rr0, rr1, lim=1.0):
        out = []
        while len(out) < k:
            u, v = rng.uniform(-1.0, 1.0, 2)
            r = rng.uniform(rr0, rr1)
            # the blob's own radius, in ellipse units, must fit inside
            e = (u * rx) ** 2 / max(rx - r * m, 1.0) ** 2 + (v * ry) ** 2 / max(ry - r * m, 1.0) ** 2
            if e <= lim:
                out.append((u, v, r, rng.uniform(-1.0, 1.0)))
        return out
    sil = sample(n, rmin, rmax, 1.0)
    leaves = sample(3 * n, leaf[0], leaf[1], 0.92)
    gaps = sample(max(2, n // 10), 0.035, 0.06, 0.45)
    fleck = sample(int(n * flecks), 0.03, 0.055, 0.75)

    def shade(u, v):
        nz = math.sqrt(max(0.05, 1.0 - u * u - v * v))
        nv = np.array([u, -v, nz])
        nv /= np.linalg.norm(nv)
        return nz, max(0.0, float(nv @ Lv))

    def tone(f, j):
        f = min(1.0, max(0.0, f))
        c = _mix(dark, mid, 2.0 * f) if f < 0.5 else _mix(mid, light, 2.0 * f - 1.0)
        return tuple(max(10, min(245, int((ch + 4 * j) * dim))) for ch in c)
    for u, v, r, j in sil:
        pygame.draw.circle(surf, (*tone(0.28, j), 255), (int(cx + u * rx), int(cy + v * ry)),
                           max(1, int(r * m)))
    blobs = []
    for u, v, r, j in leaves:
        nz, sh = shade(u, v)
        blobs.append((nz, u, v, r, sh, j))
    blobs.sort(key=lambda b: b[0])
    for nz, u, v, r, sh, j in blobs:
        pygame.draw.circle(surf, (*tone(0.16 + 0.70 * sh, j), 255),
                           (int(cx + u * rx), int(cy + v * ry)), max(1, int(r * m)))
    for u, v, r, j in gaps:
        pygame.draw.circle(surf, (*tone(0.12, j), 255),
                           (int(cx + u * rx), int(cy + (0.2 + 0.5 * abs(v)) * ry)),
                           max(1, int(r * m)))
    for u, v, r, j in fleck:
        nz, sh = shade(u, v)
        if sh < 0.62:
            continue
        pygame.draw.circle(surf, (*tone(0.70 + 0.22 * sh, j), 255),
                           (int(cx + u * rx), int(cy + v * ry)), max(1, int(r * m)))


def _conifer(surf, rng, cx, W, H, top, bot, n_t, cols, light):
    """Tiers of drooping branches: each a triangle whose lower edge is a
    saw of branch tips and whose flanks sag a little, split into a lit and a
    shaded half, with short dark needle strokes in the lower half. Lower
    tiers are darker (the crown shades itself)."""
    dark, mid, lightc = cols
    for i in range(n_t):
        f = i / (n_t - 1)
        yt = top + (bot - top) * max(0.0, f - 0.26)
        yb = top + (bot - top) * min(1.0, f + 0.14)
        hw = 0.5 * W * (0.12 + 0.86 * f) * rng.uniform(0.86, 1.04)
        apex = (cx + rng.uniform(-0.02, 0.02) * W, yt)
        teeth = 12 + int(8 * f)
        low = []
        for k in range(teeth + 1):
            q = k / teeth
            x = cx - hw + 2 * hw * q
            edge = abs(q - 0.5) * 2.0
            droop = 0.035 * H * (0.4 + 0.6 * edge)
            y = yb + (droop if k % 2 == 0 else -0.25 * droop) + rng.uniform(-0.008, 0.008) * H
            low.append((x + rng.uniform(-0.01, 0.01) * W, y))

        def flank(p0, p1, sag):
            pts = []
            for k in range(1, 4):
                q = k / 4
                pts.append((p0[0] + (p1[0] - p0[0]) * q + sag * rng.uniform(0.5, 1.0),
                            p0[1] + (p1[1] - p0[1]) * q + rng.uniform(-0.01, 0.01) * H))
            return pts
        lf = flank(apex, low[0], 0.02 * W)
        rf = flank(apex, low[-1], -0.02 * W)
        mid_i = teeth // 2
        left = [apex] + lf + low[:mid_i + 1]
        right = [apex] + low[mid_i:] + rf[::-1]
        base = 0.45 + 0.55 * (1.0 - f) ** 0.8
        jit = rng.uniform(-0.05, 0.05)
        if light == 'L':
            cl, cr = _mix(mid, lightc, 0.62 * (1.1 - base) + jit), _mix(dark, mid, 0.55 * (1.1 - base) + jit)
        elif light == 'K':
            cl = cr = _mix(dark, mid, 0.30 * (1.1 - base) + jit)
        else:
            cl = cr = _mix(dark, lightc, 0.52 * (1.1 - base) + 0.1 + jit)
        pygame.draw.polygon(surf, (*cl, 255), left)
        pygame.draw.polygon(surf, (*cr, 255), right)
        # needles: short strokes from the tier's axis out and down
        sk = _mix(dark, (0, 0, 0), 0.15)
        for _ in range(int(10 + 16 * f)):
            q = rng.uniform(-0.9, 0.9)
            y0 = yt + (yb - yt) * rng.uniform(0.45, 0.9)
            x0 = cx + q * hw * (y0 - yt) / max(yb - yt, 1.0)
            pygame.draw.line(surf, (*sk, 255), (x0, y0),
                             (x0 + 0.05 * W * q, y0 + 0.02 * H), max(1, int(0.006 * H)))
        if light == 'K':    # the rim the sun draws round a backlit crown:
            # the outer ends of the flanks only (the rest is under the tier above)
            rim = _mix(mid, lightc, 0.45)
            for seg in ([low[0], lf[2], lf[1]], [rf[1], rf[2], low[-1]]):
                pygame.draw.lines(surf, (*rim, 255), False, seg, max(1, int(0.006 * H)))


def _trunk(surf, cx, y0, y1, w, light, dim=1.0):
    lit = tuple(int(c * 1.18 * dim) for c in C_TRUNK)
    shd = tuple(int(c * 0.72 * dim) for c in C_TRUNK)
    half = max(1, int(w / 2))
    if light == 'L':
        cl, cr = lit, shd
    elif light == 'K':
        cl = cr = shd
    else:
        cl = cr = tuple(int(c * 0.98 * dim) for c in C_TRUNK)
    pygame.draw.rect(surf, (*cl, 255), (int(cx - half), int(y1), half, max(1, int(y0 - y1))))
    pygame.draw.rect(surf, (*cr, 255), (int(cx), int(y1), half, max(1, int(y0 - y1))))


def _paint_tree(sp: str, shape: int, light: str) -> pygame.Surface:
    """One master sprite, TREE_MASTER_H tall, trunk base TREE_FOOT up from
    the bottom, a soft contact shadow round it."""
    cfg = SPECIES[sp]
    H = TREE_MASTER_H
    W = max(8, int(round(cfg['aspect'] * H)))
    cols = cfg['cols']
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    surf.fill((*cols[0], 0))
    rng = np.random.default_rng(zlib.crc32(f'{sp}:{shape}'.encode()))
    L = _LIGHT_VEC[light]
    yb = H * (1.0 - TREE_FOOT)
    sh = (32, 46, 30, 215)
    cx = 0.5 * W
    if sp == 'broad':
        pygame.draw.ellipse(surf, sh, (cx - 0.36 * W, yb - 0.022 * H, 0.72 * W, 0.044 * H))
        _trunk(surf, cx, yb, 0.55 * H, 0.07 * W, light)
        _canopy(surf, rng, cx, 0.41 * H, 0.48 * W, 0.37 * H, 34, 0.18, 0.32, cols, L)
    elif sp == 'poplar':
        pygame.draw.ellipse(surf, sh, (cx - 0.45 * W, yb - 0.018 * H, 0.9 * W, 0.036 * H))
        _trunk(surf, cx, yb, 0.8 * H, 0.12 * W, light)
        _canopy(surf, rng, cx, 0.44 * H, 0.48 * W, 0.42 * H, 30, 0.40, 0.62, cols, L,
                leaf=(0.16, 0.26), flecks=0.4)
    elif sp == 'bush':
        pygame.draw.ellipse(surf, sh, (cx - 0.46 * W, yb - 0.05 * H, 0.92 * W, 0.1 * H))
        _canopy(surf, rng, cx, 0.52 * H, 0.47 * W, 0.40 * H, 26, 0.22, 0.40, cols, L)
    elif sp == 'copse':
        pygame.draw.ellipse(surf, sh, (cx - 0.46 * W, yb - 0.025 * H, 0.92 * W, 0.05 * H))
        subs = []
        for k in range(4 + shape):
            u = rng.uniform(-0.62, 0.62)
            hh = rng.uniform(0.62, 1.0)
            subs.append((rng.uniform(0, 1), u, hh))
        subs.sort()
        for depth, u, hh in subs:
            ccx = cx + u * 0.5 * W
            rx = 0.22 * W * rng.uniform(0.8, 1.15)
            ry = 0.36 * H * hh
            cy = yb - 0.12 * H - ry
            _trunk(surf, ccx, yb - 0.01 * H, cy + 0.6 * ry, 0.025 * W, light, dim=0.9)
            _canopy(surf, rng, ccx, cy, rx, ry, 18, 0.2, 0.36, cols, L, dim=0.84 + 0.16 * depth)
    else:   # conifer
        pygame.draw.ellipse(surf, sh, (cx - 0.42 * W, yb - 0.02 * H, 0.84 * W, 0.04 * H))
        _trunk(surf, cx, yb, 0.84 * H, 0.09 * W, light)
        _conifer(surf, rng, cx, W, H, 0.03 * H, 0.86 * H, 9 + shape, cols, light)
    return surf


class _TreeArt:
    """Every tree sprite, built once per PROCESS (they depend on nothing but
    the display format): masters per (species, shape, light) with a mip
    chain, then per size bucket (TREE_B0 * TREE_RATIO^k px tall) and haze
    bucket a colour-keyed RLE surface. 'R' is 'L' mirrored. The unhazed
    buckets up to TREE_PREBUILD px are made here; everything else on first
    use and kept (the big ones FIFO, TREE_CACHE_BIG of them).

    A haze SLOT is hb + HAZE_STEPS * ab: hb the haze bucket (hb /
    (HAZE_STEPS - 1) of LAND_HAZE), ab the fade step -- surface alpha
    255 * (1 - ab / FADE_STEPS) on the same RLE surface, for a tree in the
    last FADE_FRAC of its range. Measured, an RLE colour-keyed blit with
    surface alpha is 0.45 us against 0.38 opaque at 40 px (0.85 / 0.65 at
    80 px): the fade is free, where fading the COLOUR to the haze left a
    pale ghost in front of the darker painted tree line.

    A sprite is addressed by ONE int (`key`), computed for every visible
    tree in a single numpy expression: the per-tree Python work of a frame
    is then a dict lookup and a blit."""

    def __init__(self):
        t0 = time.perf_counter()
        self.masters = {}
        self.keyed = {}
        self.cache = {}
        self.big = {}
        self.aspect = np.array([SPECIES[s]['aspect'] for s in SPECIES_ORDER])
        self.nk = int(math.floor(math.log(TREE_PX_MAX / TREE_B0) / math.log(TREE_RATIO))) + 2
        self.heights = TREE_B0 * TREE_RATIO ** np.arange(self.nk)
        self.nshape = max(SHAPES.values())
        for si, sp in enumerate(SPECIES_ORDER):
            for shp in range(SHAPES[sp]):
                for lt in ('F', 'L', 'K'):
                    m = _paint_tree(sp, shp, lt)
                    mips = [m]
                    while mips[-1].get_height() > 40:
                        q = mips[-1]
                        mips.append(pygame.transform.smoothscale(
                            q, (max(1, q.get_width() // 2), max(1, q.get_height() // 2))))
                    self.masters[(si, shp, lt)] = mips
        self._warm_fl = set()
        for k in range(self.nk):
            if self.heights[k] > TREE_PREBUILD:
                break
            for (si, shp, lt) in list(self.masters):
                for l_ in ((0,) if lt == 'F' else (1, 2) if lt == 'L' else (3,)):
                    self.fetch(self.key(si, shp, l_, k, 0))
        self.build_ms = (time.perf_counter() - t0) * 1e3

    def key(self, si, shp, lt, k, hb):
        return (((si * self.nshape + shp) * 4 + lt) * self.nk + k) * NSLOT + hb

    def keys(self, si, shp, lt, k, hb) -> np.ndarray:
        return (((si * self.nshape + shp) * 4 + lt) * self.nk + k) * NSLOT + hb

    def _unkey(self, key):
        hb = key % NSLOT
        key //= NSLOT
        k = key % self.nk
        key //= self.nk
        lt = key % 4
        key //= 4
        return key // self.nshape, key % self.nshape, lt, k, hb

    def _keyed(self, mk):
        """A master thresholded to the colour key once: a NEAREST scale of it
        keeps every key pixel exact, so a big bucket needs no mask pass."""
        e = self.keyed.get(mk)
        if e is None:
            m = self.masters[mk][0]
            e = pygame.mask.from_surface(m, 127).to_surface(setsurface=m,
                                                            unsetcolor=(*KEY, 255))
            if pygame.display.get_surface() is not None:
                e = e.convert()
            self.keyed[mk] = e
        return e

    def _base(self, si, shp, lt, k):
        """The unhazed bucket: smoothscaled from the nearest mip, thresholded
        to a colour key by a pygame Mask (all C: ~9 us), with pixel (0, 0)
        forced to the key -- the haze variants read their key back there.
        Taller than the master (a tree beside the camera, only off the road)
        it is a nearest upscale of the keyed master instead (TREE_BIG_PER_FRAME)."""
        mk = (si, shp, 'L' if lt in (1, 2) else ('F' if lt == 0 else 'K'))
        mips = self.masters[mk]
        h = max(2, int(round(self.heights[k])))
        w = max(2, int(round(self.aspect[si] * h)))
        if h > TREE_MASTER_H:
            out = pygame.transform.scale(self._keyed(mk), (w, h))
            if lt == 2:
                out = pygame.transform.flip(out, True, False)
            out.set_at((0, 0), KEY)
            out.set_colorkey(KEY, pygame.RLEACCEL)
            return out
        src = mips[0]
        for q in mips:
            if q.get_height() >= h:
                src = q
        s = pygame.transform.smoothscale(src, (w, h))
        if lt == 2:
            s = pygame.transform.flip(s, True, False)
        out = pygame.mask.from_surface(s, 127).to_surface(setsurface=s,
                                                           unsetcolor=(*KEY, 255))
        if pygame.display.get_surface() is not None:
            out = out.convert()
        out.set_at((0, 0), KEY)
        out.set_colorkey(KEY, pygame.RLEACCEL)
        return out

    def fetch(self, key):
        """(surface, half width, height above the trunk base) for a key."""
        e = self.cache.get(key)
        if e is not None:
            return e
        e = self.big.get(key)
        if e is not None:
            return e
        si, shp, lt, k, slot = self._unkey(int(key))
        hb, ab = slot % HAZE_STEPS, slot // HAZE_STEPS
        if slot == 0:
            s = self._base(si, shp, lt, k)
        elif ab > 0:
            # a fade step: the hazed surface, drawn with surface alpha
            b = self.fetch(key - HAZE_STEPS * ab)[0]
            s = b.copy()
            s.set_colorkey(b.get_colorkey()[:3], pygame.RLEACCEL)
            s.set_alpha(int(round(255.0 * (1.0 - ab / FADE_STEPS))), pygame.RLEACCEL)
        else:
            # haze = two SDL fills on a copy (multiply by 1 - h, add h * HAZE):
            # ~14 us, exact to one level. The key pixel is transformed with
            # everything else, so the new key is read back from (0, 0).
            b = self.fetch(key - hb)[0]
            s = b.copy()
            h = LAND_HAZE * hb / (HAZE_STEPS - 1.0)
            m = int(round((1.0 - h) * 255.0))
            s.fill((m, m, m), special_flags=pygame.BLEND_RGB_MULT)
            s.fill(tuple(int(round(c * h)) for c in HAZE), special_flags=pygame.BLEND_RGB_ADD)
            s.set_colorkey(s.get_at((0, 0))[:3], pygame.RLEACCEL)
        w, hh = s.get_size()
        e = (s, 0.5 * w, hh * (1.0 - TREE_FOOT))
        if self.heights[k] <= TREE_PREBUILD:
            self.cache[key] = e
        else:
            if len(self.big) >= TREE_CACHE_BIG:
                self.big.pop(next(iter(self.big)))
            self.big[key] = e
        return e

    def warm(self, fl: float, slot_fn) -> float:
        """Build every (bucket, haze) pair a tree of its species' height
        range CAN show at this focal length, so the first lap does not pay for
        them a frame at a time. `slot_fn(depths, r_tree)` is the frame's own
        slot formula. Returns ms. Once per focal length per process.

        The FADE steps (ab > 0) are left to first use: prebuilt, they were
        1,872 of the 4,792 sprites built here (66.2 MB of pixels -> 58.0),
        most never drawn; peak RSS over the review's four-lap sweep (on the
        road, 60 m off both sides, 45 m off) 433 -> 407 MB, 271 -> 250 MB
        after the Renderer is built. One is a copy of its warmed hazed
        sprite plus a surface alpha, and only a tree in the last FADE_FRAC
        of its range (352-430 m; 282-344 on 'low': the 6-86 px buckets)
        asks for one: a lap of the arena builds ~1,000, 53 of them in its
        first frame, and the props' worst frame of that lap is 3.4 ms (3.5
        prebuilt)."""
        key_fl = int(round(fl))
        if key_fl in self._warm_fl:
            return 0.0
        t0 = time.perf_counter()
        self._warm_fl.add(key_fl)
        for si, sp in enumerate(SPECIES_ORDER):
            lo, hi = SPECIES[sp]['h']
            for k in range(self.nk):
                px = self.heights[k]
                if px > TREE_PREBUILD:
                    break
                d0 = fl * lo * TREE_IMG_K / (px * TREE_RATIO ** 0.5)
                d1 = fl * hi * TREE_IMG_K / (px / TREE_RATIO ** 0.5)
                if d0 > R_TREE or d1 < TREE_NEAR:
                    continue
                dd = np.linspace(max(d0, TREE_NEAR), min(d1, R_TREE), 17)
                slots = set()
                for r_t in (R_TREE, R_TREE * TREE_LOW_R):
                    sl = slot_fn(dd[dd < r_t], r_t)
                    slots.update(sl[sl >= 0].tolist())
                # every fade step's hazed sprite, not the step itself
                slots = {sl % HAZE_STEPS for sl in slots}
                for shp in range(SHAPES[sp]):
                    for lt in range(4):
                        for sl in sorted(slots):
                            self.fetch(self.key(si, shp, lt, k, sl))
        return (time.perf_counter() - t0) * 1e3

    def fetch_soft(self, key, budget):
        """fetch(), but a BIG bucket not yet built costs one of this frame's
        `budget` builds; when none are left, the nearest built bucket of the
        same tree, light and haze stands in (it is 15 % off in size for the
        frame or two until the budget reaches it)."""
        e = self.big.get(key)
        if e is not None:
            return e
        si, shp, lt, k, hb = self._unkey(int(key))
        if self.heights[k] <= TREE_PREBUILD or budget[0] > 0:
            if self.heights[k] > TREE_PREBUILD:
                budget[0] -= 1
            return self.fetch(key)
        for dk in (-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6):
            if 0 <= k + dk < self.nk:
                kk = key + dk * NSLOT
                e = self.big.get(kk) or self.cache.get(kk)
                if e is not None:
                    return e
        return self.fetch(key)

    def get(self, si, shp, lt, k, hb):
        return self.fetch(self.key(si, shp, {'F': 0, 'L': 1, 'R': 2, 'K': 3}[lt], k, hb))[0]


_ART = None
_CACHE = {}


def _tree_art() -> _TreeArt:
    global _ART
    if _ART is None:
        _ART = _TreeArt()
    return _ART


def _track_key(tr) -> tuple:
    xy = np.ascontiguousarray(tr.xy, dtype=np.float64)
    return (str(getattr(tr, 'name', '')), len(xy), round(float(tr.length), 6),
            float(tr.width), zlib.crc32(xy.tobytes()),
            len(getattr(tr, 'areas', []) or []), tuple(getattr(tr, 'guide_radii', []) or []))


def _layout_geo(tr):
    """(Layout, Geo), cached per track geometry: a Renderer is built per
    session and per self-check, and the layout is a pure function of the
    track."""
    key = _track_key(tr)
    hit = _CACHE.get(key)
    if hit is None:
        lay = Layout(tr)
        lay.edge = _Edge(tr)
        t0 = time.perf_counter()
        geo = _build_geo(lay)
        _attach_trees(geo, lay)
        lay.geo_ms = (time.perf_counter() - t0) * 1e3
        if len(_CACHE) > 8:
            _CACHE.pop(next(iter(_CACHE)))
        hit = _CACHE[key] = (lay, geo)
    return hit


def _attach_trees(g, lay) -> None:
    n = len(lay.xy)
    g.t_xy = lay.xy
    g.t_P3 = np.column_stack([lay.xy, np.zeros(n)])
    g.t_h = lay.h
    g.t_himg = lay.h * TREE_IMG_K
    g.t_r = lay.r
    g.t_sp = lay.sp
    g.t_shape = lay.shape
    g.t_detail = lay.detail
    # plan colours, a little variation per tree
    rng = np.random.default_rng(zlib.crc32((lay.name + 'tone').encode()))
    j = rng.uniform(-0.08, 0.08, n)
    mids = np.array([SPECIES[s]['cols'][1] for s in SPECIES_ORDER], dtype=np.float64)
    lits = np.array([SPECIES[s]['cols'][2] for s in SPECIES_ORDER], dtype=np.float64)
    can = np.clip(mids[lay.sp] * (1.0 + j[:, None]), 0, 255).astype(np.int32)
    hil = np.clip(0.5 * (mids[lay.sp] + lits[lay.sp]) * (1.0 + j[:, None]), 0, 255).astype(np.int32)
    g.t_can = [tuple(c) for c in _guard(can).tolist()]
    g.t_hil = [tuple(c) for c in _guard(hil).tolist()]
    # the shadow of the crown centre (0.6 h up) along -SUN_DIR
    g.t_sh = lay.xy - (0.6 * lay.h)[:, None] * SUN_XY[None, :]
    # the chase view's ground shadow: the crown (centre ~0.55 h up) cast down
    # the sun ray, an ellipse stretched 1.35x along it (the sun is 45 deg up)
    sh_h = SUN[:2] / np.linalg.norm(SUN[:2])
    perp = np.array([-sh_h[1], sh_h[0]])
    cen = lay.xy - (0.55 * lay.h)[:, None] * SUN_XY[None, :]
    th = np.linspace(0.0, 2.0 * math.pi, TREE_SHADOW_N, endpoint=False)
    ra, rb = 1.35 * lay.r, 0.95 * lay.r
    g.t_shp = (cen[:, None, :] - (ra[:, None] * np.cos(th)[None, :])[:, :, None] * sh_h
               + (rb[:, None] * np.sin(th)[None, :])[:, :, None] * perp)
    # a shadow is opaque grass-in-shade: never let one reach the run-off
    if n:
        e_c = lay.edge.exact(cen, cap=RUNOFF_MAX + 30.0)
        g.t_shok = (e_c - np.maximum(ra, 0.6 * lay.h) >= RUNOFF_MAX + 0.5) \
            | (e_c >= RUNOFF_MAX + 30.0)
    else:
        g.t_shok = np.zeros(0, dtype=bool)
    sun_h = SUN[:2] / np.linalg.norm(SUN[:2])
    g.t_hl = lay.xy + (0.3 * lay.r)[:, None] * sun_h[None, :]


# ======================================================================= #
#  THE RENDERER HOOK                                                      #
# ======================================================================= #
class Props:
    """Built once per Renderer (per track). `draw` is the only per-frame
    entry; it keeps its own timing (`last_ms`, both passes of the frame)."""

    def __init__(self, rnd):
        self.track = rnd.track
        self.lay, self.geo = _layout_geo(rnd.track)
        self.art = None
        if pygame.display.get_init() and pygame.display.get_surface() is not None:
            self.art = _tree_art()
        # the haze curves (_haze_tables): GROUND (low props, shadows) and
        # LAND (tall props, trees)
        self._hz_d, self._hz_h, self._hz_l = _haze_tables()
        # the camera's clip planes (render.Chase3D.set_fov's)
        rm = _render_mod()
        self._zn = float(rm.CHASE_Z_NEAR)
        self._cmargin = 1.0 + float(rm.CHASE_NEAR_MARGIN)
        # a tree's haze slot by depth, tabulated (one index a frame, not
        # interp + rint + fade on every visible tree)
        self._slot_tab = {lw: self._tree_slot(np.arange(0.0, r_t, 1.0 / SLOT_PER_M), r_t)
                          for lw, r_t in ((False, R_TREE), (True, R_TREE * TREE_LOW_R))}
        self.warm_ms = 0.0
        if self.art is not None:
            self.warm_ms = self.art.warm(rnd._chase.fl, self._tree_slot)
        self._far = []
        self._near = []
        self._items = None
        self.last_ms = 0.0
        self.min_depth = INF        # nearest drawn vertex depth, this frame
        self.n_drawn = (0, 0)       # (polygons, sprites) this frame
        self.n_clip = (0, 0)        # (clipped, over MAX_CLIP and dropped)
        self._frame_t0 = 0.0

    # ------------------------------------------------------------------ #
    def draw(self, rnd, x: float, y: float, psi: float, far: bool) -> None:
        """far=True: in the chase view every prop deeper than the car
        (`rnd._car_depth`), in a plan view every prop. far=False: chase only,
        the props between the eye and the car."""
        t0 = time.perf_counter()
        if rnd._cam3 is not None:
            if far:
                self.min_depth = INF
                self._prep_chase(rnd, x, y, psi)
                self._paint(rnd, self._far)
            else:
                self._paint(rnd, self._near)
        elif far:
            self._draw_plan(rnd)
        dt = (time.perf_counter() - t0) * 1e3
        self.last_ms = dt if far else self.last_ms + dt

    # ------------------------------------------------------------------ #
    def _hz(self, d, land=None):
        """Haze at camera depths d: the ground curve, or where `land` is
        True the land curve."""
        hg = np.interp(d, self._hz_d, self._hz_h)
        if land is None:
            return hg
        return np.where(land, np.interp(d, self._hz_d, self._hz_l), hg)

    @staticmethod
    def _fade(d, rmax):
        """0 .. 1 over the last FADE_FRAC of a draw range: how far faded out."""
        return np.clip((d - (1.0 - FADE_FRAC) * rmax) / (FADE_FRAC * rmax), 0.0, 1.0)

    def _tree_slot(self, d, r_tree) -> np.ndarray:
        """_TreeArt haze slot of a tree at depth d in a range r_tree: its land
        haze bucket and its fade step; -1 where it has faded out."""
        d = np.asarray(d, dtype=np.float64)
        hb = np.clip(np.rint(np.interp(d, self._hz_d, self._hz_l) / LAND_HAZE
                             * (HAZE_STEPS - 1)), 0, HAZE_STEPS - 1).astype(np.int64)
        ab = np.rint(self._fade(d, r_tree) * FADE_STEPS).astype(np.int64)
        return np.where(ab < FADE_STEPS, hb + HAZE_STEPS * ab, -1)

    def _clip(self, Qp, budget, fl, W2, H2):
        """A camera-space polygon (or a segment: 2 vertices) clipped to the
        camera's own frustum -- render.Chase3D's planes: the near plane at
        CHASE_Z_NEAR and the four screen edges widened by CHASE_NEAR_MARGIN --
        and projected: (list of int (x, y) screen points, list of depths), or
        None. Pure Python (_clip_cam): 4.9 us for a 6-gon against
        Chase3D.clip_poly's 48.5 (a numpy pass per plane is overhead at six
        vertices). `budget` [left, refused] caps the calls at MAX_CLIP."""
        if budget[0] <= 0:
            budget[1] += 1
            return None
        budget[0] -= 1
        C = _clip_cam([tuple(q) for q in Qp.tolist()], self._zn,
                      W2 * self._cmargin / fl, H2 * self._cmargin / fl)
        if len(C) < (2 if len(Qp) == 2 else 3):
            return None
        return ([(int(W2 + fl * x / z), int(H2 - fl * y / z)) for x, y, z in C],
                [z for _x, _y, z in C])

    def _car_rect(self, x, y, psi, eye, B, fl, W2, H2):
        """The car's screen bounding box (4 x 2 corners) from its body box
        (render.car_geom), or None when a corner is at or behind the near
        plane (then every near occluder is taken to overlap it)."""
        try:
            cg = _render_mod().car_geom()
            xr, xf, hw, hz = float(cg.x_rear), float(cg.x_front), float(cg.half_w), float(cg.height)
        except Exception:                  # no car fitted: a generic hatch
            xr, xf, hw, hz = -2.1, 2.1, 0.9, 1.45
        c, s = math.cos(psi), math.sin(psi)
        P = np.array([(x + c * a - s * b, y + s * a + c * b, zz)
                      for zz in (0.0, hz) for a in (xr, xf) for b in (-hw, hw)])
        Q = (P - eye) @ B.T
        if Q[:, 2].min() <= self._zn:
            return None
        sx = W2 + fl * Q[:, 0] / Q[:, 2]
        sy = H2 - fl * Q[:, 1] / Q[:, 2]
        x0, x1, y0, y1 = sx.min(), sx.max(), sy.min(), sy.max()
        return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def _prep_chase(self, rnd, x: float, y: float, psi: float) -> None:
        c3 = rnd._cam3
        g = self.geo
        eye = c3.eye
        B = np.vstack([c3._r, c3._u, c3._f])
        fl = c3.fl
        W2, H2 = 0.5 * c3.W, 0.5 * c3.H
        kx = W2 / fl * FRUSTUM_PAD
        ky = H2 / fl * FRUSTUM_PAD
        low = getattr(rnd.cfg, 'detail', 'high') == 'low'
        budget = [MAX_CLIP, 0]
        # 1. objects
        Oc = (g.o_c - eye) @ B.T
        of = Oc[:, 2]
        rad = g.o_rad
        ofp = np.maximum(of, 0.0)
        vis = ((of + rad > 0.0) & (of - rad < g.o_rmax)
               & (np.abs(Oc[:, 0]) - rad < ofp * kx)
               & (np.abs(Oc[:, 1]) - rad < ofp * ky))
        if low:
            vis &= ~g.o_detail
        # 2. polygons: object verdict, LOD window, backface
        po = g.p_obj
        od = of[po]
        lo, hi = (g.p_lo_l, g.p_hi_l) if low else (g.p_lo, g.p_hi)
        pm = vis[po] & (od >= lo) & (od < hi)
        if low:
            pm &= g.p_kind != 2          # detail 'low': no alpha shadows
        pi = np.flatnonzero(pm)
        if len(pi):
            fac = g.p_two[pi] | (np.einsum('ij,ij->i', g.p_nrm[pi], eye - g.p_cen[pi]) > 0.0)
            pi = pi[fac]
        # 3. project the survivors in one call. A polygon wholly deeper than
        #    NEAR_Z and inside the clamp band is projected directly -- a
        #    pinhole maps it exactly; the few that reach nearer (beside or
        #    behind the eye) or past the band go through the camera's own
        #    clipper instead of being dropped (a 110 m pit wall beside the
        #    camera, or the whole shadow the camera sits in, used to vanish)
        #    or clamped (a clamped vertex bends every edge that meets it).
        cn = g.p_cnt[pi]
        tot = int(cn.sum())
        ns = np.cumsum(cn) - cn
        extra = []                       # (position in pi, screen pts, depths)
        if tot:
            vidx = np.repeat(g.p_st[pi] - ns, cn) + np.arange(tot)
            Q = (g.V[vidx] - eye) @ B.T
            z = Q[:, 2]
            zmin = np.minimum.reduceat(z, ns)
            zmax = np.maximum.reduceat(z, ns)
            zsum = np.add.reduceat(z, ns)
            zz = np.maximum(z, NEAR_Z)
            sx = W2 + fl * Q[:, 0] / zz
            sy = H2 - fl * Q[:, 1] / zz
            # direct: every vertex deeper than NEAR_Z and inside the band
            bad = ((z <= NEAR_Z) | (np.abs(Q[:, 0]) * fl > zz * (W2 + CLAMP_PX))
                   | (np.abs(Q[:, 1]) * fl > zz * (H2 + CLAMP_PX)))
            direct = ~np.logical_or.reduceat(bad, ns)
            rest = np.flatnonzero(~direct)
            if len(rest):
                # every vertex beyond the same screen edge (the edge planes
                # pass through the eye, so this holds behind it too): off
                # screen, no clip. The exact edges, not the cull's padded
                # ones -- a dragstrip wall 60-190 px left of the frame was
                # three wasted clips a frame
                ex, ey = (W2 + 1.0) / fl, (H2 + 1.0) / fl
                rc = cn[rest]
                r0 = np.cumsum(rc) - rc
                rv = np.repeat(ns[rest] - r0, rc) + np.arange(int(rc.sum()))
                q = Q[rv]
                code = ((q[:, 0] >= q[:, 2] * ex).astype(np.int8)
                        | ((q[:, 0] <= -q[:, 2] * ex).astype(np.int8) << 1)
                        | ((q[:, 1] >= q[:, 2] * ey).astype(np.int8) << 2)
                        | ((q[:, 1] <= -q[:, 2] * ey).astype(np.int8) << 3)
                        | ((q[:, 2] <= 0.0).astype(np.int8) << 4))
                rest = rest[np.bitwise_and.reduceat(code, r0) == 0]
                for j in rest[np.argsort(zmin[rest])].tolist():
                    c = self._clip(Q[ns[j]:ns[j] + cn[j]], budget, fl, W2, H2)
                    if c is not None:
                        extra.append((j, c[0], c[1]))
            pts = _pairs(sx, sy)
            keep = np.flatnonzero(direct)
            pd = zsum[keep] / cn[keep]
            pz, pzx = zmin[keep], zmax[keep]     # each polygon's depth range
            if len(keep):
                self.min_depth = float(zmin[keep].min())
            ns, cn = ns[keep], cn[keep]
            if extra:
                n0 = len(pts)
                ej = np.array([e[0] for e in extra], dtype=np.int64)
                ec = np.array([len(e[1]) for e in extra], dtype=np.int64)
                for e in extra:
                    pts += e[1]
                es = n0 + np.cumsum(ec) - ec
                keep = np.concatenate([keep, ej])
                ns = np.concatenate([ns, es])
                cn = np.concatenate([cn, ec])
                pd = np.concatenate([pd, [sum(e[2]) / len(e[2]) for e in extra]])
                pz = np.concatenate([pz, [min(e[2]) for e in extra]])
                pzx = np.concatenate([pzx, [max(e[2]) for e in extra]])
                self.min_depth = min(self.min_depth, min(min(e[2]) for e in extra))
            pi = pi[keep]
        else:
            pts, pd = [], np.zeros(0)
            pz = pzx = pd
        K = len(pi)
        obj = g.p_obj[pi]
        odk = of[obj]
        # 3b. which side of the car each polygon is painted on. An object
        #     wholly nearer or deeper than the car goes with its centre. For
        #     one that STRADDLES the car's depth the centre can be wrong by
        #     the object's size: a pits face 0.8-2.0 m from the eye, its
        #     building centred 8.9 m deep behind a car at 7.4, was painted
        #     before the car, which then stood in front of a wall that was
        #     in front of it; a trailer side running -0.3..9.5 m past the
        #     car (centre 3.8 m) was painted after it and cut across it. So
        #     a face of a straddling object is painted after the car exactly
        #     when its plane separates the eye from the car (a plane the car
        #     is on the eye's side of cannot hide any of it); a line (no
        #     plane) goes by its own depth range, else its object's centre
        cd = float(rnd._car_depth)
        lay_p = g.p_layer[pi]
        orad = rad[obj]
        near = (lay_p != 0) & (odk <= cd)
        strad = np.flatnonzero((lay_p != 0) & (odk - orad < cd) & (odk + orad > cd))
        if len(strad):
            car = np.array([x, y, 0.7])          # the car body's mid-height
            pj = pi[strad]
            nr, cen = g.p_nrm[pj], g.p_cen[pj]
            sep = (np.einsum('ij,ij->i', nr, eye - cen)
                   * np.einsum('ij,ij->i', nr, car - cen)) < 0.0
            by_depth = np.where(pzx[strad] < cd, True,
                                np.where(pz[strad] > cd, False, near[strad]))
            near[strad] = np.where(g.p_kind[pj] == 1, by_depth, sep)
        # 3c. near-camera occluders. Clipped at the near plane instead of
        #     dropped at NEAR_Z, a sponsor board 0.35-1 m in front of the
        #     eye covered 1,018,578 of the 1,024,000 px2 frame and hid the
        #     car. A polygon painted after the car that reaches nearer than
        #     Z_OCC AND overlaps the car's screen box fades out (alpha) to
        #     nothing at Z_GONE -- the car shows through, and nothing pops.
        #     A wall or a shadow running beside or past the camera is not
        #     between the eye and the car (3b / ground) and stays clipped.
        a_occ = np.ones(K)
        occ = np.flatnonzero(near & (pz < Z_OCC))
        if len(occ):
            rect = self._car_rect(x, y, psi, eye, B, fl, W2, H2)
            for j in occ.tolist():
                q = np.array(pts[ns[j]:ns[j] + cn[j]], dtype=np.float64)
                if rect is None or _sat(q, rect):
                    a_occ[j] = min(1.0, max(0.0, (pz[j] - Z_GONE) / (Z_OCC - Z_GONE)))
        # 4. haze per polygon by its object's depth: the land curve for a
        #    tall object, the ground curve for a low one and for shadows;
        #    over the last FADE_FRAC of its range a polygon fades OUT (alpha),
        #    a line (draw.line has no alpha; a far pole is 1 px) into what
        #    stands behind it: a tall object's into the haze of the sky, a
        #    low one's into the ground's tone past LAND_D (FOOT). Into the
        #    haze colour, the low fence rails at 375 m were drawn (184, 198,
        #    211) on grass of (129, 159, 125): pale lines on the horizon
        kinds = g.p_kind[pi].copy()
        land = g.o_land[obj] & (g.p_layer[pi] != 0)
        h = self._hz(odk, land)
        f = self._fade(odk, g.o_rmax[obj])
        base = g.p_col[pi]
        rgb = base + (HAZE[None, :] - base) * h[:, None]
        ln = np.flatnonzero((kinds == 1) & (f > 0.0))
        if len(ln):
            tgt = np.where(land[ln, None], HAZE[None, :], FOOT[None, :])
            rgb[ln] += (tgt - rgb[ln]) * f[ln, None]
        rgb = _guard(np.rint(rgb).astype(np.int32))
        sh = kinds == 2
        fa = (kinds == 0) & ((f > 0.0) | (a_occ < 1.0))
        alpha = np.rint(np.where(sh, SHADOW_ALPHA * (1.0 - h), 255.0) * (1.0 - f)
                        * a_occ).astype(np.int32)
        # a faded occluder too faint to see is not drawn at all (a line has
        # no alpha: it goes at half way)
        gone = (a_occ < 1.0) & np.where(kinds == 1, a_occ < 0.5, alpha < 8)
        rgb[sh] = 0
        kinds[fa] = 2
        cols = _quads(rgb, alpha)
        wpx = np.maximum(1, np.rint(g.p_w[pi] * fl / np.maximum(pd, NEAR_Z))).astype(int).tolist()
        # trees: cull, size bucket, haze slot, light by bearing to the sun
        sprites = []
        tz_list = np.zeros(0)
        if self.art is not None and len(g.t_xy):
            TQ = (g.t_P3 - eye) @ B.T
            tz = TQ[:, 2]
            tzp = np.maximum(tz, 1e-3)
            r_tree = R_TREE * (TREE_LOW_R if low else 1.0)
            tv = ((tz > TREE_NEAR) & (tz < r_tree)
                  & (np.abs(TQ[:, 0]) - g.t_r < tzp * kx)
                  & (TQ[:, 1] < tzp * ky) & (TQ[:, 1] + g.t_himg > -tzp * ky))
            if low:
                tv &= ~g.t_detail
            ti = np.flatnonzero(tv)
            px = fl * g.t_himg[ti] / tz[ti]
            tab = self._slot_tab[low]
            slot = tab[np.minimum((tz[ti] * SLOT_PER_M).astype(np.int64), len(tab) - 1)]
            okp = (px >= 2.0) & (px <= TREE_PX_MAX) & (slot >= 0)
            ti, px, slot = ti[okp], px[okp], slot[okp]
            tz_list = tz[ti]
            kb = np.clip(np.rint(np.log(px / TREE_B0) / math.log(TREE_RATIO)), 0,
                         self.art.nk - 1).astype(int)
            vx = g.t_xy[ti, 0] - eye[0]
            vy = g.t_xy[ti, 1] - eye[1]
            vl = np.maximum(np.hypot(vx, vy), 1e-6)
            sh = SUN[:2] / np.linalg.norm(SUN[:2])
            cosang = (vx * sh[0] + vy * sh[1]) / vl
            crs = (vx * sh[1] - vy * sh[0]) / vl
            lt = np.where(cosang < -0.45, 0, np.where(cosang > 0.45, 3,
                                                      np.where(crs > 0.0, 1, 2)))
            sxp = W2 + fl * TQ[ti, 0] / tz_list
            syp = H2 - fl * TQ[ti, 1] / tz_list
            keys = self.art.keys(g.t_sp[ti], g.t_shape[ti], lt, kb, slot).tolist()
            cget = self.art.cache.get
            fetch = self.art.fetch_soft
            tbud = [TREE_BIG_PER_FRAME]
            for key, x_, y_ in zip(keys, sxp.tolist(), syp.tolist()):
                e = cget(key)
                if e is None:
                    e = fetch(key, tbud)
                sprites.append((e[0], (int(x_ - e[1]), int(y_ - e[2]))))
            # their cast shadows, near trees only (ground layer), fading out
            # over the last FADE_FRAC of r_sh; one reaching the eye is clipped
            r_sh = TREE_SHADOW_R * (DETAIL_LOW_R if low else 1.0)
            si_ = ti[(tz_list < r_sh) & g.t_shok[ti]]
            if len(si_):
                n_s = len(si_)
                V3 = np.concatenate([g.t_shp[si_], np.full((n_s, TREE_SHADOW_N, 1), 0.02)],
                                    axis=2)
                SQ = ((V3.reshape(-1, 3) - eye) @ B.T).reshape(n_s, TREE_SHADOW_N, 3)
                sz = SQ[:, :, 2]
                zc = np.maximum(sz, NEAR_Z)
                ssx = W2 + fl * SQ[:, :, 0] / zc
                ssy = H2 - fl * SQ[:, :, 1] / zc
                okz = ((sz.min(axis=1) > NEAR_Z)
                       & (np.abs(ssx - W2) < W2 + CLAMP_PX).all(axis=1)
                       & (np.abs(ssy - H2) < H2 + CLAMP_PX).all(axis=1))
                n0 = len(pts)
                pts = pts + _pairs(ssx[okz].ravel(), ssy[okz].ravel())
                s_st = (n0 + TREE_SHADOW_N * np.arange(int(okz.sum()))).tolist()
                s_cn = [TREE_SHADOW_N] * len(s_st)
                d_s = tz[si_[okz]].tolist()
                for j in np.flatnonzero(~okz & (sz.max(axis=1) > 0.0)).tolist():
                    c = self._clip(SQ[j], budget, fl, W2, H2)
                    if c is None:
                        continue
                    s_st.append(len(pts))
                    s_cn.append(len(c[0]))
                    pts += c[0]
                    d_s.append(float(tz[si_[j]]))
                    self.min_depth = min(self.min_depth, min(c[1]))
                ksh = len(s_st)
                if ksh:
                    d_s = np.array(d_s)
                    hs = self._hz(d_s)
                    fs = self._fade(d_s, r_sh)
                    sc_ = _quads(_guard(np.rint(np.array(SHADOW_RGB)[None, :] * (1.0 - hs[:, None])
                                                + HAZE[None, :] * hs[:, None]).astype(np.int32)),
                                 np.rint(255.0 * (1.0 - fs)).astype(np.int32))
                    sk = np.where(fs > 0.0, 2, 0).astype(np.int64)
                    kinds = np.concatenate([kinds, sk])
                    cols = cols + sc_
                    wpx = wpx + [1] * ksh
                    ns = np.concatenate([ns, np.array(s_st, dtype=np.int64)])
                    cn = np.concatenate([cn, np.array(s_cn, dtype=np.int64)])
                    odk = np.concatenate([odk, d_s])
                    pd = np.concatenate([pd, d_s])
                    K += ksh
        self.n_clip = (MAX_CLIP - budget[0], budget[1])
        T = len(sprites)
        # 5. one painter's order: ground first, objects far -> near, decals
        #    after their face, polygons far -> near; trees merged in
        n_sh = K - len(pi)
        layer = np.concatenate([g.p_layer[pi], np.zeros(n_sh, dtype=np.int64),
                                np.ones(T, dtype=np.int64)])
        okey = np.concatenate([odk, tz_list])
        sub = np.concatenate([g.p_sub[pi], np.zeros(n_sh, dtype=np.int64),
                              np.zeros(T, dtype=np.int64)])
        pkey = np.concatenate([pd, tz_list])
        order = np.lexsort((-pkey, sub, -okey, layer))
        # 6. split at the car (3b): ground and deeper-than-the-car first
        farm = np.concatenate([~near, np.ones(n_sh, dtype=bool), tz_list > cd])
        if gone.any():
            order = order[~np.concatenate([gone, np.zeros(n_sh + T, dtype=bool)])[order]]
        fm = farm[order]
        self._far = order[fm].tolist()
        self._near = order[~fm].tolist()
        st = (np.array(ns, dtype=np.int64)).tolist()
        self._items = (K, kinds.tolist(), cols, st, cn.tolist(), wpx, pts, sprites)
        self._drawn_pi = pi
        self.n_drawn = (K, T)

    def _paint(self, rnd, lst) -> None:
        if not lst or self._items is None:
            return
        K, kinds, cols, st, cn, wpx, pts, sprites = self._items
        sc = rnd.screen
        poly = pygame.draw.polygon
        line = pygame.draw.line
        apoly = pygame.gfxdraw.filled_polygon
        blit = sc.blit
        for j in lst:
            if j < K:
                k = kinds[j]
                s0 = st[j]
                if k == 0:
                    poly(sc, cols[j], pts[s0:s0 + cn[j]])
                elif k == 1:
                    line(sc, cols[j], pts[s0], pts[s0 + 1], wpx[j])
                else:
                    apoly(sc, pts[s0:s0 + cn[j]], cols[j])
            else:
                s, p = sprites[j - K]
                blit(s, p)

    # ------------------------------------------------------------------ #
    def _draw_plan(self, rnd) -> None:
        """Footprints: shadows, strokes, roofs and tops, tree circles."""
        g = self.geo
        cx, cy = float(rnd.cam[0]), float(rnd.cam[1])
        r = rnd._view_radius() * 1.02
        low = getattr(rnd.cfg, 'detail', 'high') == 'low'
        dx = g.o_pc[:, 0] - cx
        dy = g.o_pc[:, 1] - cy
        vis = dx * dx + dy * dy < (r + g.o_prad) ** 2
        if low:
            vis &= ~g.o_detail
        qi = np.flatnonzero(vis[g.q_obj])
        tv = (g.t_xy[:, 0] - cx) ** 2 + (g.t_xy[:, 1] - cy) ** 2 < (r + g.t_r + 0.6 * g.t_h) ** 2
        if low:
            tv &= ~g.t_detail
        ti = np.flatnonzero(tv)
        cn = g.q_cnt[qi]
        tot = int(cn.sum())
        ns = np.cumsum(cn) - cn
        nt = len(ti)
        parts = []
        if tot:
            parts.append(g.q_V[np.repeat(g.q_st[qi] - ns, cn) + np.arange(tot)])
        if nt:
            parts += [g.t_xy[ti], g.t_sh[ti], g.t_hl[ti]]
        if not parts:
            return
        SS = rnd.world_to_screen(np.vstack(parts))
        S = _pairs(SS[:, 0], SS[:, 1])
        ppm = float(rnd.ppm)
        wpx = np.maximum(g.q_min[qi], np.rint(g.q_w[qi] * ppm)).astype(int).tolist()
        sc = rnd.screen
        poly = pygame.draw.polygon
        line = pygame.draw.line
        circ = pygame.draw.circle
        apoly = pygame.gfxdraw.filled_polygon
        kinds = g.q_kind
        cols = g.q_col
        layers = g.q_layer[qi].tolist()
        st = ns.tolist()
        cnl = cn.tolist()
        qil = qi.tolist()
        tr_px = np.maximum(1, np.rint(g.t_r[ti] * ppm)).astype(int).tolist()
        hl_px = np.maximum(1, np.rint(0.55 * g.t_r[ti] * ppm)).astype(int).tolist()
        o_c, o_s, o_h = tot, tot + nt, tot + 2 * nt
        trees_done = [False, False]

        # zoomed far out (the manual zoom goes to 2.1 px/m, a 360 m radius
        # and most of a map's trees in view: 1.75 ms measured) the highlight
        # is 1-3 px and the shadow a smudge: one circle a tree, like 'low'
        thin = low or ppm < PLAN_THIN_PPM

        def trees(stage):
            # detail 'low': half the trees (t_detail) and one circle each
            if stage == 0:
                if not thin:
                    shok = g.t_shok[ti].tolist()
                    for k in range(nt):
                        if shok[k]:
                            circ(sc, SHADOW_RGB, S[o_s + k], tr_px[k])
            elif thin:
                for k, i in enumerate(ti.tolist()):
                    circ(sc, g.t_can[i], S[o_c + k], tr_px[k])
            else:
                for k, i in enumerate(ti.tolist()):
                    circ(sc, g.t_can[i], S[o_c + k], tr_px[k])
                    if tr_px[k] >= 3:
                        circ(sc, g.t_hil[i], S[o_h + k], hl_px[k])
        for j, q in enumerate(qil):
            lay_ = layers[j]
            if lay_ >= 1 and not trees_done[0]:
                trees(0)
                trees_done[0] = True
            if lay_ >= 3 and not trees_done[1]:
                trees(1)
                trees_done[1] = True
            k = kinds[q]
            s0 = st[j]
            if k == 0:
                poly(sc, cols[q], S[s0:s0 + cnl[j]])
            elif k == 1:
                line(sc, cols[q], S[s0], S[s0 + 1], wpx[j])
            elif k == 2:
                circ(sc, cols[q], S[s0], max(1, wpx[j]))
            elif not low:
                apoly(sc, S[s0:s0 + cnl[j]], (0, 0, 0, SHADOW_ALPHA))
        if not trees_done[0]:
            trees(0)
        if not trees_done[1]:
            trees(1)


# ======================================================================= #
#  SELF-CHECK                                                             #
# ======================================================================= #
#: what each map must contain (the check fails if a theme silently lost one)
_CIRCUIT_KINDS = ('pillar', 'gantry', 'pits', 'tower', 'stand', 'tyres', 'armco', 'board',
                  'fence', 'brake', 'marshal', 'flood', 'flag', 'trailer')
REQUIRED = {
    #  every circuit: the three after the arena were shaped for the full set
    #  (a pit straight >= 120 m through s = 0, infields wide enough for the
    #  paddock, R <= 65 corners for the boards and a stand)
    'arena': _CIRCUIT_KINDS,
    'linden': _CIRCUIT_KINDS,
    'kestrel': _CIRCUIT_KINDS,
    'ashdown': _CIRCUIT_KINDS,
    'fairfield': _CIRCUIT_KINDS,
    'open': ('hangar', 'tower', 'office', 'windsock', 'fence', 'cones', 'flood', 'car'),
    'skidpad': ('office', 'flood', 'windsock', 'cones'),
    'dragstrip': ('wall', 'xmas', 'timing', 'gatepost', 'stand', 'tower', 'lightpole'),
    'generated-test': ('pillar', 'gantry', 'pits', 'stand', 'tyres', 'armco', 'board'),
    'hairpin-test': ('pits', 'stand', 'tyres', 'armco'),
}


def _drawn_extent(rnd) -> float:
    """How far past the frame the props' drawn screen points reach, px."""
    pr = rnd._props
    if pr is None or pr._items is None:
        return 0.0
    K, kinds, cols, st, cn, wpx, pts, sprites = pr._items
    lst = pr._far + pr._near
    idx = [j for j in lst if j < K]
    if not idx:
        return 0.0
    P = np.array([q for j in idx for q in pts[st[j]:st[j] + cn[j]]], dtype=np.float64)
    return float(max(-P[:, 0].min(), P[:, 0].max() - rnd.W, -P[:, 1].min(),
                     P[:, 1].max() - rnd.H, 0.0))


def clearance_report(lay) -> dict:
    """{class: (min edge distance m, min area distance m, n)} over every
    ground footprint (sampled every 0.5 m, exact distance), and 'tree' over
    the canopies (trunk distance minus canopy radius). The gantry's beam is
    OVERHEAD, not a ground footprint: its pillars are measured instead."""
    E = getattr(lay, 'edge', None) or _Edge(lay.tr)
    out = {}
    for p in lay.props:
        if p.cls == 'overhead':
            continue
        P = _outline(p.foot, 0.5)
        e = float(E.exact(P).min())
        a = float(E.area(P).min())
        m = out.get(p.cls, (INF, INF, 0))
        out[p.cls] = (min(m[0], e), min(m[1], a), m[2] + 1)
    if len(lay.xy):
        e = E.exact(lay.xy) - lay.r
        a = E.area(lay.xy) - lay.r
        out['tree'] = (float(e.min()), float(a.min()), int(len(lay.xy)))
    return out


def _gen_track():
    """A circuit the layout was never tuned for, under a name no theme knows
    (so the circuit theme runs on it): an asymmetric loop, 60 m and 40 m
    corners, a 240 m start straight; it closes by construction."""
    from . import track as trk
    segs = [trk.Seg.straight(240.0), trk.Seg.arc(60.0, 90.0), trk.Seg.straight(80.0),
            trk.Seg.arc(40.0, 90.0), trk.Seg.straight(280.0), trk.Seg.arc(40.0, 90.0),
            trk.Seg.straight(80.0), trk.Seg.arc(60.0, 90.0)]
    tr = trk.Track(name='generated-test', width=11.0, segs=segs, origin=(0.0, 0.0),
                   heading0=0.0, closed=True, sector_s=[0.0])
    return trk.build(tr)


def _hairpin_track():
    """Two 12 m hairpins joined by 120 m straights, 9 m wide: round a
    hairpin the barrier line 31 m out is four times the centreline's length
    -- the case that put a barrier chord 29.3 m from the edge."""
    from . import track as trk
    S, A = trk.Seg.straight, trk.Seg.arc
    tr = trk.Track(name='hairpin-test', width=9.0, segs=[S(120.0), A(12.0, 180.0), S(120.0),
                                                        A(12.0, 180.0)],
                   origin=(0.0, 0.0), heading0=0.0, closed=True, sector_s=[0.0])
    return trk.build(tr)


def self_check(verbose: bool = True, screenshot_dir: str = 'runs') -> bool:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    from . import render as R_
    from . import track as trk
    results = []

    def rep(tag, ok, msg):
        results.append((tag, bool(ok), msg))
        if verbose:
            print(f"  [{'ok' if ok else 'FAIL'}] {tag:46s} {msg}")

    if verbose:
        print('drive/props.py self-check')
    names = trk.TRACK_ORDER
    tracks = {n: trk.make_track(n) for n in names}
    tracks['generated-test'] = _gen_track()
    tracks['hairpin-test'] = _hairpin_track()

    # ---- 1. determinism: the same track is the same layout, bit for bit
    for n, tr in tracks.items():
        a, b = Layout(tr), Layout(tr)
        rep(f'{n}: deterministic layout', a.digest() == b.digest(),
            f'digest {a.digest()} twice, built in {a.build_ms:.0f} ms')

    # ---- 2. clearance, counts, the required classes, the palette
    for n, tr in tracks.items():
        lay, geo = _layout_geo(tr)
        rp = clearance_report(lay)
        bad = []
        for cls, (e, a, k) in sorted(rp.items()):
            need = CLEAR_SOLID if cls == 'tree' else CLEAR[cls]
            if e < need or a < CLEAR_AREA:
                bad.append(f'{cls} {e:.2f} m')
        txt = ', '.join(f'{c} {e:.1f} m' + (f'/area {a:.0f}' if a < INF else '')
                        for c, (e, a, k) in sorted(rp.items()))
        rep(f'{n}: clearance per class (min m)', not bad,
            ('VIOLATED: ' + ', '.join(bad) + '; ') * bool(bad) + txt)
        rep(f'{n}: nothing on the ribbon', all(v[0] > 0.0 for v in rp.values()),
            f'nearest footprint {min(v[0] for v in rp.values()):.2f} m past the edge')
        # opaque (grass-in-shade) shadows only past the run-off band
        op = [i for i in range(geo.n_poly) if geo.p_layer[i] == 0 and geo.p_kind[i] == 0]
        dmin = INF
        for i in op:
            V = geo.V[geo.p_st[i]:geo.p_st[i] + geo.p_cnt[i], :2]
            dmin = min(dmin, float(lay.edge.exact(_outline(V, 2.0), cap=RUNOFF_MAX + 2.0).min()))
        rep(f'{n}: opaque shadows beyond the run-off', dmin >= RUNOFF_MAX,
            f'{len(op)} opaque, nearest {dmin:.1f} m from the edge (>= {RUNOFF_MAX:.0f}); '
            f'{int(((geo.p_layer == 0) & (geo.p_kind == 2)).sum())} alpha')
        cnt = lay.counts()
        need = REQUIRED.get(n, ())
        miss = [k for k in need if cnt.get(k, 0) == 0]
        rep(f'{n}: counts per class', n in REQUIRED and not miss and cnt.get('tree', 0) > 200,
            ('NO REQUIRED ENTRY; ' * (n not in REQUIRED))
            + ('MISSING ' + ','.join(miss) + '; ') * bool(miss)
            + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items()))
            + f'; soup {geo.n_poly} polys, {geo.n_obj} objects')
        fb = int(((np.rint(geo.p_col).astype(np.int32)[:, None, :]
                   == FORBIDDEN[None]).all(-1)).sum())
        fq = sum(1 for c in geo.q_col if tuple(c[:3]) in {tuple(f) for f in FORBIDDEN.tolist()})
        rep(f'{n}: no self-check colour in the palette', fb == 0 and fq == 0,
            f'{fb} soup polygons, {fq} plan items')

    # ---- 3. both views, every map, through a headless Renderer: the props'
    #         own time per frame, the near plane, the colours drawn
    calib = R_.cpu_calibration()
    os.makedirs(screenshot_dir, exist_ok=True)
    all_min_depth = INF
    max_px = 0.0
    ratios = []
    forbid = {tuple(f) for f in FORBIDDEN.tolist()}
    for n in names:
        tr = tracks[n]
        L = float(tr.length)
        for mode in ('chase', 'car_up'):
            times = {}
            for detail in ('high', 'low'):
                rnd = R_.Renderer(R_.ViewConfig(mode=mode, detail=detail), tr, headless=True)
                props = rnd._props
                ts = []
                forb = 0
                n_st = 24
                for k in range(n_st):
                    s = (k + 0.5) * L / n_st if tr.closed else 5.0 + k * (L - 10.0) / n_st
                    i = R_.trk_index(tr, s)
                    for j, (dn, dpsi) in enumerate(((0.0, 0.0), (0.0, 0.0), (4.0, 0.5))):
                        p_ = float(tr.psi[i]) + dpsi
                        st = R_._demo_state(float(tr.xy[i][0]) - dn * math.sin(p_),
                                            float(tr.xy[i][1]) + dn * math.cos(p_), p_, u=25.0)
                        rnd.update_camera(st, 1.0 / 60.0 if j else 0.0)
                        rnd.draw_frame(st, None, 0.0, R_._demo_ctl(), R_._demo_hud(V=25.0),
                                       R_.SkidBuffer())
                        ts.append(props.last_ms)
                        if mode == 'chase':
                            all_min_depth = min(all_min_depth, props.min_depth)
                            max_px = max(max_px, _drawn_extent(rnd))
                            if props._items is not None:
                                forb += sum(1 for k_, c in zip(props._items[1], props._items[2])
                                            if k_ != 2 and tuple(c[:3]) in forbid)
                    if k == n_st // 3 and detail == 'high':
                        rnd.screenshot(os.path.join(os.path.abspath(screenshot_dir),
                                                    f'props_{n}_{mode}.png'))
                times[detail] = np.array(ts[3:])
                if mode == 'chase' and detail == 'high':
                    rep(f'{n}: chase draws no self-check colour', forb == 0,
                        f'{forb} polygons over {len(ts)} frames')
            th, tl = times['high'], times['low']
            bud = (2.5, 6.0) if mode == 'chase' else (1.5, 4.0)
            ok, why = R_.frame_budget_verdict(float(th.mean()), float(np.percentile(th, 99)),
                                              budget_mean=bud[0], budget_p99=bud[1], calib=calib)
            ratios.append(float(tl.mean() / max(th.mean(), 1e-6)))
            rep(f'{n}: {mode} props time', ok,
                f'{why}; detail low {tl.mean():.2f} ms (x{ratios[-1]:.2f})')
    # the collector: the props' per-frame screen lists must not drive it
    # (see _pairs: ndarray.tolist() of (N, 2) took gen-0 passes 24 -> 297 and
    # gen-2 stalls 0 -> 2 x 4.3 ms over 300 frames)
    import gc
    tr = tracks['arena']
    gcn = {}
    for on in (False, True):
        rnd = R_.Renderer(R_.ViewConfig(mode='chase'), tr, headless=True)
        if not on:
            rnd._props = None
        gc.collect()
        c0 = gc.get_stats()[0]['collections']
        for k in range(150):
            i = R_.trk_index(tr, 12.0 + 0.5 * k)
            st = R_._demo_state(float(tr.xy[i][0]), float(tr.xy[i][1]), float(tr.psi[i]), u=30.0)
            rnd.update_camera(st, 1.0 / 60.0 if k else 0.0)
            rnd.draw_frame(st, None, 0.0, R_._demo_ctl(), R_._demo_hud(V=30.0), R_.SkidBuffer())
        gcn[on] = gc.get_stats()[0]['collections'] - c0
    rep('chase: the props do not drive the collector', gcn[True] - gcn[False] <= 15,
        f'gen-0 collections over 150 arena frames: {gcn[False]} without, {gcn[True]} with')
    # the near plane: a polygon reaching past the eye is CLIPPED, not dropped
    # (the review measured up to 740 000 px2 of a frame missing 40 m off the
    # arena -- a pit wall, a stand, the shadow the camera sat in) and not
    # clamped (a clamped vertex bends its edges). Off the road, sideways and
    # backwards, where the camera stands among the props:
    ncl, nref, tmax = 0, 0, 0.0
    for tname, offs in (('arena', (40.0, -40.0, 60.0, -33.0)), ('dragstrip', (-33.0, 40.0)),
                        ('open', (-40.0,))):
        tr = tracks[tname]
        rnd = R_.Renderer(R_.ViewConfig(mode='chase'), tr, headless=True)
        L = float(tr.length)
        for n_off in offs:
            for dpsi in (0.0, 0.5 * math.pi, math.pi):
                for s in np.arange(4.0, L, L / 14.0):
                    i = R_.trk_index(tr, float(s))
                    p_ = float(tr.psi[i])
                    st = R_._demo_state(float(tr.xy[i][0]) - n_off * math.sin(p_),
                                        float(tr.xy[i][1]) + n_off * math.cos(p_), p_ + dpsi,
                                        u=25.0)
                    rnd.update_camera(st, 0.0)
                    rnd.draw_frame(st, None, 0.0, R_._demo_ctl(), R_._demo_hud(V=25.0),
                                   R_.SkidBuffer())
                    pr = rnd._props
                    ncl += pr.n_clip[0]
                    nref += pr.n_clip[1]
                    tmax = max(tmax, pr.last_ms)
                    all_min_depth = min(all_min_depth, pr.min_depth)
                    max_px = max(max_px, _drawn_extent(rnd))
    rep('chase: polygons past the eye are clipped', ncl > 0 and nref == 0,
        f'{ncl} clipped, {nref} refused over the off-road sweep (MAX_CLIP {MAX_CLIP} a '
        f'frame); worst props frame there {tmax:.1f} ms')
    rep('chase: nothing drawn behind the near plane', all_min_depth >= 0.999 * R_.CHASE_Z_NEAR,
        f'nearest drawn vertex {all_min_depth:.2f} m (camera near plane {R_.CHASE_Z_NEAR} m)')
    rep('chase: drawn vertices are exact projections', max_px < CLAMP_PX,
        f'nothing is clamped; the furthest drawn vertex is {max_px:.0f} px past the '
        f'frame (past {CLAMP_PX} a polygon goes through the clipper instead)')
    rep("detail 'low' is cheaper", float(np.mean(ratios)) < 0.85,
        f'mean low / high {np.mean(ratios):.2f} over {len(ratios)} map x view pairs')

    # ---- 4. the far land: the tree belts end in the world's painted tree
    #         line -- the same airiness, then an alpha fade (nothing pops)
    art = _tree_art()
    k40 = int(np.argmin(np.abs(art.heights - 40.0)))
    mc = []
    for si, sp in enumerate(SPECIES_ORDER):
        for shp in range(SHAPES[sp]):
            sf = art.fetch(art.key(si, shp, 0, k40, 0))[0]
            a = pygame.surfarray.array3d(sf).reshape(-1, 3).astype(np.float64)
            mc.append(a[~(a == np.array(sf.get_colorkey()[:3])).all(axis=1)].mean(axis=0))
    mc = np.mean(mc, axis=0)

    def lum(c):
        return float(np.dot(np.asarray(c, dtype=np.float64), (0.2126, 0.7152, 0.0722)))
    h_f = float(_land_haze(_world.haze_factor((1.0 - FADE_FRAC) * R_TREE)))
    far = mc + (HAZE - mc) * h_f
    lo_l, hi_l = lum(_world.TREE_LINE), lum(_world.TREE_LINE_BASE)
    pr = Props.__new__(Props)
    pr._hz_d, pr._hz_h, pr._hz_l = _haze_tables()
    sl = pr._tree_slot(np.linspace((1.0 - FADE_FRAC) * R_TREE, R_TREE - 1e-6, 400), R_TREE)
    ab_last = int(sl[sl >= 0][-1]) // HAZE_STEPS
    rep('far trees fade into the painted tree line', lo_l <= lum(far) <= hi_l
        and int(sl[-1]) == -1 and ab_last == FADE_STEPS - 1,
        f'mean tree hazed {h_f:.2f} at the fade start = lum {lum(far):.0f}, painted line '
        f'{lo_l:.0f} (crowns) .. {hi_l:.0f} (foot); alpha steps down to '
        f'{255 * (1 - ab_last / FADE_STEPS):.0f} before the range ends')

    # ---- 5. the sprites: no self-check colour, the key never leaks out
    fb, n_px = 0, 0
    for key, (surf, _hw, _h) in list(art.cache.items())[::5]:
        a = pygame.surfarray.array3d(surf).reshape(-1, 3).astype(np.int32)
        fb += int(((a[:, None, :] == FORBIDDEN[None]).all(-1)).sum())
        n_px += len(a)
    rep('tree sprites: no self-check colour', fb == 0,
        f'{fb} of {n_px} px sampled; {len(art.cache)} sprites cached, built in '
        f'{art.build_ms:.0f} ms (once per process)')

    ok = all(r[1] for r in results)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'}: {sum(r[1] for r in results)}/{len(results)} checks")
    return ok


if __name__ == "__main__":
    sys.exit(0 if self_check() else 1)
