"""drive/world.py -- what the world looks like round the road.

The backdrop (sky, the distant land, the ground under everything), the
ground dressing beside the road (verge, run-off, gravel), the detail ON the
tarmac, and the atmosphere (haze toward the horizon). Drawn by the hooks in
`render.Renderer.draw_frame`, and -- when a World is built -- by the track
layers of render.py, which hand their work here (`World.ribbon`, `.kerbs`,
`.lines`, ...). The layout it draws comes from `drive/scenery.py`.

Read-only with respect to physics, like the rest of the renderer: it reads
the track and the renderer's camera, never the vehicle. It does not import
`render` at module level (render imports it); it is handed the renderer,
and render's constants are read through `sys.modules['drive.render']`.

Why it looks the way it does -- the cost model, measured on this machine
(pygame 2.5.2, dummy driver, 1280x800, load ~6):
  * `pygame.draw.polygon` costs VERTICES x ROWS, not area: an 800-vertex
    strip 470 rows tall is 0.90 ms, the same strip at 200 vertices 0.18 ms,
    and a hundred small quads 0.06 ms together. So long strips (ribbon,
    verge, edge lines, rubber) are DECIMATED by curvature and depth before
    they are drawn (`keep_masks`), and small things are drawn freely.
  * what made the chase view slow was never pygame: it was ~100 numpy
    clipper calls a frame, one per centre dash and kerb block (dashes alone
    6-10 ms). Every layer here projects ALL its vertices in ONE numpy pass
    (`project_polys`, `project_segs`); the few polygons that straddle the
    eye go through Chase3D's exact clipper, the rest are drawn directly.
    A polygon with every vertex in front of the eye projects exactly --
    a pinhole maps a segment in front of it to a segment -- so the fast
    path is not an approximation.
  * an opaque 1280x400 blit is 0.07 ms and a 1280x40 SRCALPHA blit 0.03 ms,
    so the sky and the distant land are ONE pre-rendered panorama per track
    (2*pi*focal-length px round, scrolled with the view's heading, built
    once when the Renderer is) and the haze is ONE pre-rendered alpha band
    keyed to the horizon row.

The haze model: in the chase view the camera has no roll, so a ground point's
camera depth is a function of its SCREEN ROW alone. `draw_atmosphere` blends
every ground row below the horizon toward HAZE_RGB by `haze_factor(depth of
that row)` in one blit, capped at HAZE_LAND where the ground runs under the
panorama's land -- the road, the grass and the kerbs fade exactly as far as
they are. It is drawn before the props and the particles, which haze
themselves by their own depth (`hazed()` / `haze_factor`).
"""
from __future__ import annotations

import math
import sys
import time
import zlib

import numpy as np
import pygame

from . import scenery as scn

# --- the shared look: drive/props.py and drive/fx.py read these, so the
# stands, the trees and the smoke fade into the SAME haze and are lit by the
# same sun as the ground and the car. Keep the names; tune the values here.
#: the sun, a unit vector in the WORLD frame (x, y, z up) -- the same one
#: render.LIGHT_DIR3 shades the car with (normalised (0.45, 0.55, 0.70))
_SUN_N = math.sqrt(0.45 ** 2 + 0.55 ** 2 + 0.70 ** 2)
SUN_DIR = (0.45 / _SUN_N, 0.55 / _SUN_N, 0.70 / _SUN_N)
#: what distant things fade to (the colour of the air at the horizon)
HAZE_RGB = (188, 202, 216)
#: grass, and the second tone of its mowing stripes
GRASS_RGB = (84, 124, 60)
GRASS2_RGB = (94, 136, 66)
HAZE_NEAR = 70.0          # m  camera depth where the haze starts
HAZE_FAR = 520.0          # m  ... and where it is at HAZE_MAX
HAZE_MAX = 0.88           # -  never quite the haze colour: silhouettes stay
LAND_D = 300.0            # m  est  how far the panorama's tree line stands:
                          #    its colours are foliage seen through ~0.4 of
                          #    haze, which is haze_factor(300 m)


def haze_factor(depth: float) -> float:
    """0 (clear) .. HAZE_MAX (almost the haze colour) at a camera depth, m."""
    f = (float(depth) - HAZE_NEAR) / (HAZE_FAR - HAZE_NEAR)
    if f <= 0.0:
        return 0.0
    return HAZE_MAX * min(f, 1.0) ** 1.15


#: the ground's haze where it meets the land: draw_atmosphere caps the band
#: at this (0.41), and the panorama's tree line fades to LAND_FOOT_RGB at its
#: foot. Uncapped, the ground's last 6-8 rows went to 0.88 -- a pale strip
#: under a tree line hazed ~0.3, so the trees floated on a fog bank. The
#: ground beyond LAND_D is what the land would hide; capped, it is the tone
#: the land's foot is, and land and ground meet in one tone.
HAZE_LAND = haze_factor(LAND_D)
LAND_FOOT_RGB = tuple(int(round(g + (h - g) * HAZE_LAND)) for g, h in
                      zip((89, 130, 63), HAZE_RGB))     # mid-stripe grass, hazed


def hazed(rgb, depth: float) -> tuple:
    """`rgb` seen through `depth` metres of air."""
    h = haze_factor(depth)
    if h <= 0.0:
        return tuple(int(c) for c in rgb)
    return tuple(int(round(c + (hc - c) * h)) for c, hc in zip(rgb, HAZE_RGB))


def _haze_factor_v(depth: np.ndarray) -> np.ndarray:
    """haze_factor, vectorised (the same curve, bit for bit in float64)."""
    f = np.clip((np.asarray(depth, dtype=np.float64) - HAZE_NEAR)
                / (HAZE_FAR - HAZE_NEAR), 0.0, 1.0)
    return np.where(f > 0.0, HAZE_MAX * f ** 1.15, 0.0)


# ======================================================================= #
#  PALETTE -- all est: a clear late-afternoon summer day at a well-kept   #
#  European circuit. Natural and readable; the car and the road stay the  #
#  strongest things on screen. None of these is one of the RGB values the #
#  self-checks count (C_YELLOW / C_GREEN / C_BAR_BRK / C_PURPLE / the PB  #
#  ghost green): `self_check` asserts that.                               #
# ======================================================================= #
SKY_HORIZON = (200, 212, 224)     # pale, a touch lighter than the haze
SKY_ZENITH = (78, 132, 202)       # the top of a 46 deg frame, not the zenith
SUN_GLOW = (255, 236, 206)
CLOUD_LIT = (250, 250, 251)
CLOUD_SHADE = (190, 199, 214)
HILL_FAR = (140, 160, 170)        # the far ridge: mostly air
HILL_FAR_BASE = (170, 185, 196)
TREE_LINE = (84, 108, 88)         # the near tree line: some air
TREE_LINE_BASE = (122, 142, 136)
VERGE_RGB = (104, 131, 68)        # the mown verge beside the edge: drier
GRAVEL_RGB = (184, 170, 136)
GRAVEL_RAKE = (162, 148, 116)
GRAVEL_EDGE = (146, 134, 104)
RUNOFF_RGB = (124, 124, 116)      # abrasive run-off: pale warm concrete ...
RUNOFF_PAINT = (66, 122, 90)      # ... banded green: reads as NOT the road
RUBBER_RGB = (50, 52, 57)         # the rubbered line, darker than C_TARMAC
GROOVE_RGB = ((33, 34, 37), (39, 41, 45), (46, 48, 52))
PATCH_RGB = ((47, 49, 54), (68, 70, 75))
PATCH_SEAL = (37, 39, 43)
CRACK_RGB = (44, 46, 51)
#: by scenery's keys: white, dim, rubber, and (task 45, round 3) the stop
#: board's amber brake marker and its checker's black
PAINT_RGB = ((230, 231, 233), (196, 198, 201), (35, 36, 39), (236, 168, 36), (26, 27, 30))
SECTOR_RGB = (206, 209, 214)
MUD_RGB = (90, 68, 42)            # a skid on grass: churned earth
GRAVEL_SKID = (138, 124, 96)
PAD_RGB = (94, 96, 95)            # the open map's pad: concrete, lighter than
                                  # the asphalt road painted round it
PAD_JOINT = (70, 72, 72)
WATER_FRINGE = (50, 57, 66)       # the damp rim round standing water
WATER_SHEEN = ((46, 62, 79), (51, 68, 87), (60, 79, 100))  # the sky in it at
                                  # 25-45 / 45-90 / > 90 m: water reflects
                                  # more at a grazing angle (Fresnel)
WATER_STREAK = (51, 68, 87)       # puddles reflecting the sky, near ...
WATER_STREAK2 = (70, 90, 113)     # ... and far
CHALK = (168, 186, 146)           # guide lines painted on grass (faded)
TUFT_RGB = ((77, 115, 55), (96, 134, 64), (97, 124, 61))    # dark / lush / dry
_TUFT_ARR = np.array(TUFT_RGB, dtype=np.float64)
DETAIL_DEPTH = 120.0     # m   chase: rake lines, trap borders are not drawn
                         #     deeper than this
CRACK_NEAR = 8.0         # m   chase: sealed cracks from here (contrast full
CRACK_FAR = 60.0         # m   by +8 m) to here (fading over the last 15 m)
TUFT_PITCH = 5.0         # m   est  one tuft per 5 m cell, jittered
TUFT_DEPTH = 45.0        # m   chase: tufts nearer than this (< 1 px beyond)
TUFT_NEAR = 6.0          # m   chase: ... and deeper than this, their contrast
TUFT_FULL = 14.0         # m   fading in to full by here: a 0.8 m clump at 4 m
                         #     was a flat 100 px hexagon, a splotch not grass
TUFT_PPM = 11.5          # px/m plan views draw every tuft only zoomed in this
                         #      far (< 16 m/s): ~600 quads at 9 px/m was ~1 ms
TUFT_SPARSE = 4          # -   ... and below it one clump in this many, darker
                         #     and larger: the world-anchored speed cue of a
                         #     zoomed-out plan view (the 20 m grid's job)
TUFT_PLAN_RGB = ((66, 102, 47), (110, 142, 72))     # sparse clumps: dark / dry
KERB_H = 0.06            # m   est  the kerb's raised face (25-50 mm real,
                         #          a touch more so it reads at 20 m)
KERB_FACE_K = 0.60       # -   est  the shaded face's brightness
KERB_FACE_DEPTH = 45.0   # m   faces beyond this are ~1 px: not drawn
STRIPE_NEAR_D = 55.0     # m   full-contrast mowing stripes out to here ...
STRIPE_FAR_D = 150.0     # m   ... half contrast to here, then the haze
PANO_ROWS = 460          # rows of sky above the horizon (27 deg at fl 942)
PANO_CACHE = 4           # panoramas kept per process (~11 MB each)
PANO_FL_TOL = 0.006      # -   a lens within 0.6 % of the panorama's (4 px at
                         #     the frame edge) blits it as it is; beyond, a
                         #     rescaled copy is made (World._pano_for_lens)
Z_DIRECT = 0.5           # m   vertices nearer than this go through the clipper
CLAMP_PX = 4000.0        # px  ... and so do any that would hit render's clamp

#: every colour the world paints ON the road surface -- render's self-check
#: counts road pixels by exact colour, and must count these too
ROAD_COLOURS = (RUBBER_RGB, PATCH_RGB[0], PATCH_RGB[1], PATCH_SEAL, CRACK_RGB,
                GROOVE_RGB[0], GROOVE_RGB[1], GROOVE_RGB[2], WATER_FRINGE,
                WATER_SHEEN[0], WATER_SHEEN[1], WATER_SHEEN[2], WATER_STREAK,
                WATER_STREAK2)
SECTOR_BAR = 0.4         # m   a sector line's length along the road (base's)
THIN_PX = 1.3            # px  a paint quad thinner than this is drawn as a line
STRIP_SPLIT_D = (8.0, 16.0, 32.0, 64.0, 128.0)   # m  chase: long strips are
                         #     cut at these camera depths (_depth_pieces)
LINE_POLY_PX = 2.5       # px  a painted line narrower than this on screen is
                         #     drawn as a polyline: pygame's fill of a sub-
                         #     pixel sliver leaves gaps (it read as dashes)


def _R():
    """The render module, for its constants (never imported at module level:
    render imports this module)."""
    m = sys.modules.get('drive.render')
    if m is None:
        from . import render as m          # noqa: F811 -- lazy, by design
    return m


# ======================================================================= #
#  BATCHED PROJECTION -- one numpy pass per layer, clip only straddlers   #
# ======================================================================= #
def ground_depth_bottom(c3) -> float:
    """Camera depth of the ground seen on the frame's bottom row (4.3 m at
    rest). A GROUND point nearer than this projects below the frame -- the
    camera has no roll, so a ground point's row depends on its depth alone
    -- which is what lets the batch projectors drop low geometry lying
    wholly nearer than it instead of sending it through the clipper."""
    fz, uz, ez = float(c3._f[2]), float(c3._u[2]), float(c3.eye[2])
    dz = fz - (0.5 * c3.H / c3.fl) * uz
    return ez / -dz if dz < -1e-9 else 0.0


LOW_Z = 0.3              # m   geometry below this height counts as ground
                         #     for the bottom-of-frame drop (kerb tops 0.06)


def _outcodes(c3, Q) -> np.ndarray:
    """(n,) bit codes of camera-space points against Chase3D's four side
    planes (1 right, 2 left, 4 top, 8 bottom). The planes pass through the
    eye, so these are half-space tests valid in front of it and behind it:
    geometry with every point outside the SAME plane is exactly what the
    clipper would return empty, and is dropped without calling it."""
    R_ = _R()
    kx = 0.5 * c3.W * (1.0 + R_.CHASE_NEAR_MARGIN) / c3.fl
    ky = 0.5 * c3.H * (1.0 + R_.CHASE_NEAR_MARGIN) / c3.fl
    px, py, pz = Q[:, 0], Q[:, 1], Q[:, 2]
    return ((px >= pz * kx).astype(np.int8) | ((px <= -pz * kx).astype(np.int8) << 1)
            | ((py >= pz * ky).astype(np.int8) << 2) | ((py <= -pz * ky).astype(np.int8) << 3))


def _as3(V):
    V = np.asarray(V, dtype=np.float64)
    if V.shape[-1] == 3:
        return V
    return np.concatenate([V, np.zeros(V.shape[:-1] + (1,))], axis=-1)


def clip_rect(P: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman of a SCREEN polygon (n,2) against a rectangle,
    one vectorised pass per side (Chase3D.clip_poly's trick in 2-D).

    Why: pygame's fill scans every row between the polygon's min and max y
    and tests every edge on each, whether or not the row is on the surface.
    A 300-vertex ribbon reaching 2000 px below the frame fills in 1.45 ms;
    clipped to the frame first, 0.16 ms. Exact for convex polygons; on a
    concave one the bridges it leaves lie ON the rectangle's border, just
    outside the frame, and cancel in the even-odd fill."""
    for ax, lim, ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        n = len(P)
        if n < 3:
            return P[:0]
        d = (P[:, ax] - lim) if ge else (lim - P[:, ax])
        ins = d >= 0.0
        if ins.all():
            continue
        if not ins.any():
            return P[:0]
        nxt = np.roll(np.arange(n), -1)
        dn = d[nxt]
        cross = ins != ins[nxt]
        den = d - dn
        t = np.where(cross, d / np.where(np.abs(den) > 1e-12, den, 1e-12), 0.0)
        I = P + t[:, None] * (P[nxt] - P)
        st = np.empty((2 * n, 2))
        st[0::2], st[1::2] = P, I
        keep = np.empty(2 * n, dtype=bool)
        keep[0::2], keep[1::2] = ins, cross
        P = st[keep]
    return P


CLIP_MARGIN = 96.0       # px  polygons reaching further above / below the
                         #     frame than this are clipped to it before they
CLIP_MIN_VERTS = 16      #     are filled -- if they have this many vertices
                         #     (a 4-6 vertex stripe gains nothing: 30 us each)


def project_polys(rnd, polys) -> list:
    """Ground (or 3-D) polygons -> [(k, screen point list)] for the ones that
    can be on screen, in k order.

    `polys` is a (K, M, 2|3) array (uniform, e.g. quads) or a list of
    (M_k, 2|3) arrays. Plan view: one world_to_screen call and a bbox cull.
    Chase: one camera transform for every vertex; a polygon whose vertices
    are all in front of the eye (pz > Z_DIRECT) and inside the clamp band is
    projected directly -- exactly, a pinhole maps a segment in front of it to
    a segment -- and only the rest go through Chase3D.poly_px (Sutherland-
    Hodgman, exact). Wholly behind the eye or wholly off screen: dropped.
    """
    if isinstance(polys, np.ndarray):
        if polys.ndim != 3 or len(polys) == 0:
            return []
        K, M = polys.shape[0], polys.shape[1]
        V = polys.reshape(K * M, polys.shape[2])
        offs = np.arange(K) * M
        lens = np.full(K, M)
    else:
        if not polys:
            return []
        lens = np.fromiter((len(p) for p in polys), dtype=np.int64, count=len(polys))
        dim = max(np.asarray(p).shape[-1] for p in polys)
        V = np.concatenate([_as3(p) if dim == 3 else np.asarray(p, dtype=np.float64)
                            for p in polys])
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
        K = len(polys)
        M = None
    W, H = rnd.W, rnd.H
    c3 = rnd._cam3
    if c3 is None:
        S = rnd.world_to_screen(V[:, :2])
        sx, sy = S[:, 0], S[:, 1]
        direct = np.ones(K, dtype=bool)
        front = direct
    else:
        V3 = _as3(V)
        Q = c3.camera(V3)
        pz = Q[:, 2]
        pzc = np.maximum(pz, Z_DIRECT)
        sx = 0.5 * W + c3.fl * Q[:, 0] / pzc
        sy = 0.5 * H - c3.fl * Q[:, 1] / pzc
        okv = (pz > Z_DIRECT) & (np.abs(sx - 0.5 * W) < CLAMP_PX) & (
            np.abs(sy - 0.5 * H) < CLAMP_PX)
        # a vertex that matters: in front of the near plane, and either
        # above the ground or deeper than the frame's bottom row
        d_bot = ground_depth_bottom(c3) - 0.5
        fr = (pz > _R().CHASE_Z_NEAR) & ((pz > d_bot) | (V3[:, 2] > LOW_Z))
        oc = _outcodes(c3, Q)
        if M is not None:
            direct = okv.reshape(K, M).all(axis=1)
            front = fr.reshape(K, M).any(axis=1)
            front &= np.bitwise_and.reduce(oc.reshape(K, M), axis=1) == 0
        else:
            direct = np.logical_and.reduceat(okv, offs)
            front = np.logical_or.reduceat(fr, offs)
            front &= np.bitwise_and.reduceat(oc, offs) == 0
    if M is not None:
        X, Y = sx.reshape(K, M), sy.reshape(K, M)
        mnx, mxx, mny, mxy = X.min(1), X.max(1), Y.min(1), Y.max(1)
    else:
        mnx, mxx = np.minimum.reduceat(sx, offs), np.maximum.reduceat(sx, offs)
        mny, mxy = np.minimum.reduceat(sy, offs), np.maximum.reduceat(sy, offs)
    onscr = (mxx >= 0) & (mnx <= W) & (mxy >= 0) & (mny <= H)
    # worth clipping only when the wasted rows cost more than the clip: the
    # fill tests every edge on every row, so many vertices x rows off frame
    big = (lens >= CLIP_MIN_VERTS) & ((mny < -CLIP_MARGIN) | (mxy > H + CLIP_MARGIN))
    S = np.stack([sx, sy], axis=1)
    sel = np.flatnonzero((direct & onscr) | (~direct & front))
    out = []
    if not len(sel):
        return out
    # the per-polygon loop runs on Python lists: numpy scalar indexing
    # (offs[k], direct[k], ...) was ~40 % of this function's own time
    Sl = S.astype(np.int32).tolist() if len(sel) * 2 > K else None
    Si = None
    offs_l, lens_l = offs.tolist(), lens.tolist()
    dir_l, big_l = direct.tolist(), big.tolist()
    for k in sel.tolist():
        o, ln = offs_l[k], lens_l[k]
        if ln < 3:
            continue                    # not a polygon (a 1-sample strip)
        if dir_l[k]:
            if big_l[k]:
                P = clip_rect(S[o:o + ln], -2.0, -2.0, W + 2.0, H + 2.0)
                if len(P) >= 3:
                    out.append((k, P.astype(np.int32).tolist()))
            elif Sl is not None:
                out.append((k, Sl[o:o + ln]))
            else:
                if Si is None:
                    Si = S.astype(np.int32)
                out.append((k, Si[o:o + ln].tolist()))
        else:
            Qc = c3.clip_poly(V3[o:o + ln])
            if len(Qc) < 3:
                continue
            P = c3.project_cam(Qc)
            if len(P) >= CLIP_MIN_VERTS and (P[:, 1].min() < -CLIP_MARGIN
                                             or P[:, 1].max() > H + CLIP_MARGIN):
                P = clip_rect(P, -2.0, -2.0, W + 2.0, H + 2.0)
            if len(P) >= 3:
                out.append((k, P.astype(np.int32).tolist()))
    return out


def project_segs(rnd, A, B):
    """Segments (n,2|3),(n,2|3) -> (Apx, Bpx) int32 (n,2) and a live mask,
    like Renderer._gsegs, but only the segments that straddle the eye go
    through Chase3D.clip_segments; the rest are one direct projection."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = len(A)
    if n == 0:
        z = np.zeros((0, 2), dtype=np.int32)
        return z, z, np.zeros(0, dtype=bool)
    W, H = rnd.W, rnd.H
    c3 = rnd._cam3
    if c3 is None:
        SA = rnd.world_to_screen(A[:, :2])
        SB = rnd.world_to_screen(B[:, :2])
        live = ~(((SA[:, 0] < 0) & (SB[:, 0] < 0)) | ((SA[:, 0] > W) & (SB[:, 0] > W))
                 | ((SA[:, 1] < 0) & (SB[:, 1] < 0)) | ((SA[:, 1] > H) & (SB[:, 1] > H)))
        return SA.astype(np.int32), SB.astype(np.int32), live
    A3, B3 = _as3(A), _as3(B)
    Q = c3.camera(np.vstack([A3, B3]))
    pz = Q[:, 2]
    pzc = np.maximum(pz, Z_DIRECT)
    sx = 0.5 * W + c3.fl * Q[:, 0] / pzc
    sy = 0.5 * H - c3.fl * Q[:, 1] / pzc
    okv = (pz > Z_DIRECT) & (np.abs(sx - 0.5 * W) < CLAMP_PX) & (np.abs(sy - 0.5 * H) < CLAMP_PX)
    ok = okv[:n] & okv[n:]
    S = np.stack([sx, sy], axis=1)
    SA, SB = S[:n].astype(np.int32), S[n:].astype(np.int32)
    live = ok.copy()
    live &= ~(((SA[:, 0] < 0) & (SB[:, 0] < 0)) | ((SA[:, 0] > W) & (SB[:, 0] > W))
              | ((SA[:, 1] < 0) & (SB[:, 1] < 0)) | ((SA[:, 1] > H) & (SB[:, 1] > H)))
    # straddlers go to the clipper -- unless the whole segment is low and
    # nearer than the frame's bottom row, where it cannot be seen at all
    d_bot = ground_depth_bottom(c3) - 0.5
    low = (A3[:, 2] <= LOW_Z) & (B3[:, 2] <= LOW_Z)
    oc = _outcodes(c3, Q)
    seen = (~low | (pz[:n] > d_bot) | (pz[n:] > d_bot)) & ((oc[:n] & oc[n:]) == 0)
    bad = np.flatnonzero(~ok & ((pz[:n] > 0.0) | (pz[n:] > 0.0)) & seen)
    if len(bad):
        ca, cb, cl = c3.clip_segments(A3[bad], B3[bad])
        SA[bad], SB[bad] = ca, cb
        live[bad] = cl
    return SA, SB, live


def _depth_px(rnd, P2) -> np.ndarray:
    """px per metre at each ground point (chase), or rnd.ppm (plan)."""
    c3 = rnd._cam3
    if c3 is None:
        return np.full(len(P2), float(rnd.ppm))
    d = c3.ground_depth(np.asarray(P2, dtype=np.float64))
    return c3.fl / np.maximum(d, 0.5)


def _unwrap(idx: np.ndarray, n: int, closed: bool) -> np.ndarray:
    """A run's (wrapped) sample indices -> monotonic unwrapped indices."""
    if not closed or len(idx) < 2:
        return idx.astype(np.int64)
    u = np.empty(len(idx), dtype=np.int64)
    u[0] = idx[0]
    u[1:] = idx[0] + np.cumsum(np.mod(np.diff(idx.astype(np.int64)), n))
    return u


def _in_windows(s, windows, L, closed, pad=0.0) -> np.ndarray:
    """Which arclengths `s` (lap-wrapped) fall in any of the unwrapped
    windows, trying +-one lap on a closed track (render's _overlaps rule)."""
    s = np.asarray(s, dtype=np.float64)
    m = np.zeros(len(s), dtype=bool)
    shifts = (-L, 0.0, L) if closed else (0.0,)
    for w0, w1 in windows:
        for k in shifts:
            m |= (s + k >= w0 - pad) & (s + k <= w1 + pad)
    return m


# ======================================================================= #
#  STRIP DECIMATION -- vertices cost rows; drop the ones nobody can see    #
# ======================================================================= #
_KEEP: dict = {}


def keep_masks(tr) -> dict:
    """{level: bool (nper,)} centreline samples to keep at a chord tolerance.

    Greedy walk: from a kept sample, step as far as the chord's sagitta
    c^2 / 8R stays under `tol` for the TIGHTEST radius in the step's reach
    (so a straight never jumps over the start of the arc after it), capped
    at `max_gap`. Levels (tolerance m, max gap m):
      'near' 0.004 / 8   chase, camera depth < 20 m (0.4 px at 10 m)
      'mid'  0.020 / 16  chase 20-70 m, and the plan view at > 12.5 px/m
      'far'  0.080 / 30  chase > 70 m (1 px at 70 m), plan at <= 12.5 px/m
    On the arena that is 2499 samples -> 1310 / 520 / 250.
    """
    key = (id(tr), len(tr.s), float(tr.length))
    ent = _KEEP.get(key)
    if ent is not None and ent[0] is tr:
        return ent[1]
    n = (len(tr.s) - 1) if tr.closed else len(tr.s)
    ds = float(tr.ds)
    kab = np.abs(np.asarray(tr.kappa[:n], dtype=np.float64))
    out = {}
    for lvl, tol, gap in (('near', 0.004, 8.0), ('mid', 0.020, 16.0), ('far', 0.080, 30.0)):
        ms = max(1, int(gap / ds))
        kp = np.concatenate([kab, kab[:ms]]) if tr.closed else np.concatenate(
            [kab, np.full(ms, kab[-1])])
        kmax = np.lib.stride_tricks.sliding_window_view(kp, ms + 1).max(axis=1)[:n]
        stride = np.clip(np.floor(np.sqrt(8.0 * tol / np.maximum(kmax, 1e-9)) / ds),
                         1, ms).astype(np.int64)
        keep = np.zeros(n, dtype=bool)
        i = 0
        st = stride.tolist()
        while i < n:
            keep[i] = True
            i += st[i]
        keep[n - 1] = True
        out[lvl] = keep
    if len(_KEEP) > 16:
        _KEEP.clear()
    _KEEP[key] = (tr, out)
    return out


# ======================================================================= #
#  THE PANORAMA -- sky, sun glow, clouds, far hills, tree line            #
# ======================================================================= #
_PANO: dict = {}


def _smooth_noise(rng, n: int, n_knots: int, lo: float, hi: float) -> np.ndarray:
    """A periodic smooth random curve over n columns (cosine-interpolated
    knots), in [lo, hi]."""
    kn = rng.uniform(lo, hi, n_knots)
    x = np.arange(n) * n_knots / n
    i0 = np.floor(x).astype(int) % n_knots
    i1 = (i0 + 1) % n_knots
    t = x - np.floor(x)
    t = 0.5 - 0.5 * np.cos(np.pi * t)
    return kn[i0] * (1 - t) + kn[i1] * t


def _edge_window(n: int, frac: float = 0.08) -> np.ndarray:
    """(n,) weights: 0 at both ends, smoothstep up to 1 over `frac` of n."""
    i = np.arange(n) + 0.5
    t = np.clip(np.minimum(i, n - i) / max(frac * n, 1.0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_panorama(name: str, theme: str, fl: float) -> dict:
    """The sky and the distant land, one full turn, pre-rendered.

    Width 2*pi*fl px (5920 at the 46 deg chase FOV), PANO_ROWS rows above the
    horizon. Column c is the world azimuth -c/fl (clockwise), so the camera
    heading psi puts column (-psi*fl) mod width at the screen centre and the
    land stays put in the world as the camera turns. Rows are elevation
    atan(px above horizon / fl). Deterministic: seeded from the track name.
    Built when the World is (60-170 ms with the layout), cached per process
    for the last PANO_CACHE (track, focal length) pairs; never rebuilt inside
    a frame -- a different lens is served by World._pano_for_lens.
    """
    key = (name, theme, int(round(fl)))
    ent = _PANO.get(key)
    if ent is not None:
        return ent
    t0 = time.perf_counter()
    rng = np.random.default_rng(zlib.crc32(('pano' + name).encode()))
    Pw = int(round(2.0 * math.pi * fl))
    R = PANO_ROWS
    y_up = (R - np.arange(R) - 0.5)                   # px above the horizon
    elev = np.arctan(y_up / fl)                       # (R,)
    sun = np.array(SUN_DIR)
    az_sun = math.atan2(sun[1], sun[0])

    # --- the sky, at 1/8 horizontal resolution (it is smooth), then scaled
    lw = max(64, Pw // 16)
    az_lo = -(np.arange(lw) + 0.5) * (Pw / lw) / fl
    rh = R // 2
    el_lo = elev.reshape(rh, -1).mean(axis=1) if R % 2 == 0 else elev[::2]
    E, A = np.meshgrid(el_lo.astype(np.float32), az_lo.astype(np.float32),
                       indexing='ij')                  # (R/2, lw)
    g = 1.0 - np.exp(-E / 0.20)
    hor = np.array(SKY_HORIZON, dtype=np.float64)
    zen = np.array(SKY_ZENITH, dtype=np.float64)
    sky = hor[None, None, :] + (zen - hor)[None, None, :] * g[..., None]
    dx, dy, dz = np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)
    cg = dx * sun[0] + dy * sun[1] + dz * sun[2]
    glow = 0.50 * np.exp((cg - 1.0) / 0.020) + 0.30 * np.exp((cg - 1.0) / 0.30)
    # warm light low on the sun's side of the sky (late afternoon)
    dA = np.cos(A - az_sun)
    warm = 0.16 * np.exp(-E / 0.09) * np.clip(dA, 0.0, 1.0) ** 2
    sg = np.array(SUN_GLOW, dtype=np.float64)
    wt = np.clip(glow + warm, 0.0, 0.85)[..., None]
    sky = sky + (sg - sky) * wt
    lo = np.clip(sky, 0, 255).astype(np.uint8)
    lo_s = pygame.surfarray.make_surface(np.ascontiguousarray(lo.transpose(1, 0, 2)))
    sky_s = pygame.transform.smoothscale(lo_s, (Pw, R))
    img = pygame.surfarray.pixels3d(sky_s)            # (Pw, R, 3) uint8, a VIEW

    def over(col_lo, col_hi, x0, alpha, colour):
        """Composite `colour` (H,W,3 or 3,) with alpha (w, h) at column x0,
        wrapping round the panorama."""
        w = alpha.shape[0]
        cols = (np.arange(x0, x0 + w)) % Pw
        a = alpha[..., None]
        sub = img[cols, col_lo:col_hi].astype(np.float32)
        img[cols, col_lo:col_hi] = np.clip(sub * (1 - a) + colour * a, 0, 255).astype(np.uint8)

    # --- clouds: a few flat-bottomed cumulus of gaussian blobs, lit from
    #     the sun's side, hazier the lower they sit. The blobs are placed
    #     in a nominal w x h box (the RNG draws, and so every map's sky, are
    #     as they were), but the box COMPOSITED is sized from the blobs'
    #     own extent (+-2.6 radii: exp(-6.8) = 0.1 %) and its alpha is
    #     windowed to 0 over its outer columns and rows. The nominal box cut
    #     the blobs off: 6 of 9 arena clouds had alpha 0.1-0.92 in their
    #     first / last column -- flat vertical cloud sides in the sky.
    theme_clouds = {'circuit': 9, 'open': 7, 'skidpad': 8, 'dragstrip': 6}.get(theme, 8)
    cloud_edge = 0.0                 # the largest alpha on any box border
    cloud_cov = np.zeros(Pw)         # the most cloud in each column
    for _ in range(theme_clouds):
        cw = float(rng.uniform(130, 360))
        ch = cw * float(rng.uniform(0.18, 0.30))
        el = float(rng.uniform(0.05, 0.30))
        yc = R - fl * math.tan(el)                       # base row
        xc = float(rng.uniform(0, Pw))
        w, h = int(cw * 1.3), int(ch * 2.2)
        x0 = int(xc - 0.5 * w)
        y0 = int(yc - h)
        if y0 < 0 or yc > R - 12:
            continue
        blobs = []
        for _b in range(int(rng.integers(5, 12))):
            bx = float(rng.uniform(0.15, 0.85)) * w
            by = h - float(rng.uniform(0.25, 0.75)) * ch
            rx = float(rng.uniform(0.12, 0.26)) * cw
            ry = rx * float(rng.uniform(0.45, 0.75))
            blobs.append((bx, by, rx, ry))
        B_ = np.array(blobs)
        base = h - 0.12 * ch                              # the flat base (local)
        # the real extent, in the nominal box's coordinates
        ex0 = int(math.floor((B_[:, 0] - 2.6 * B_[:, 2]).min()))
        ex1 = int(math.ceil((B_[:, 0] + 2.6 * B_[:, 2]).max()))
        ey0 = max(int(math.floor((B_[:, 1] - 2.6 * B_[:, 3]).min())), -y0)
        ey1 = int(math.ceil(base))
        xs = np.arange(ex0, ex1)[:, None].astype(np.float64)
        ys = np.arange(ey0, ey1)[None, :].astype(np.float64)
        D = np.zeros((ex1 - ex0, ey1 - ey0))
        for bx, by, rx, ry in blobs:
            D += np.exp(-((xs - bx) ** 2 / rx ** 2 + (ys - by) ** 2 / ry ** 2))
        D *= np.clip((base - ys) / (0.25 * ch), 0.0, 1.0)     # the flat base
        al = np.clip(D * 1.4 - 0.30, 0.0, 1.0) ** 0.8 * 0.92
        # a smoothstep window over the outer 8 % of the box, both sides and
        # the top (the base is already flat): whatever the blobs do, the
        # border is exactly 0 -- no step where two boxes overlap
        nx_, ny_ = al.shape
        al *= _edge_window(nx_)[:, None]
        wy = _edge_window(ny_)
        wy[ny_ // 2:] = 1.0
        al *= wy[None, :]
        cloud_edge = max(cloud_edge, float(al[0].max()), float(al[-1].max()),
                         float(al[:, 0].max()))
        cc_ = (np.arange(x0 + ex0, x0 + ex1)) % Pw
        cloud_cov[cc_] = np.maximum(cloud_cov[cc_], al.max(axis=1))
        shade = np.clip((ys - (h - 1.4 * ch)) / (1.4 * ch), 0.0, 1.0)[..., None]
        az_c = -(xc / fl)
        lit = 0.5 + 0.5 * math.cos(az_c - az_sun)
        top = np.array(CLOUD_LIT, dtype=np.float64)
        bot = np.array(CLOUD_SHADE, dtype=np.float64) * (0.94 + 0.06 * lit)
        colr = top + (bot - top) * shade
        hz = math.exp(-el / 0.10) * 0.55
        colr = colr + (hor - colr) * hz
        colr = np.broadcast_to(colr, al.shape + (3,))
        over(y0 + ey0, y0 + ey1, x0 + ex0, al, colr)

    # --- the land: a remote range (palest), a far ridge, and a near tree
    #     line of round crowns and pointed poplars with gaps where fields
    #     are. Profiles are px above the horizon at fl 942, per theme:
    #     (ridge lo, ridge hi, tree lo, tree hi, remote range hi)
    prof = {'circuit': (12, 44, 5, 14, 70), 'open': (6, 22, 4, 10, 40),
            'skidpad': (9, 32, 4, 12, 55), 'dragstrip': (3, 12, 3, 8, 26)
            }.get(theme, (10, 40, 5, 12, 60))
    s = fl / 942.0
    remote = (_smooth_noise(rng, Pw, 7, 0.25, 1.0) * 0.7
              + _smooth_noise(rng, Pw, 23, 0.0, 1.0) * 0.3) * prof[4] * s
    far = _smooth_noise(rng, Pw, 11, prof[0], prof[1]) * 0.6 \
        + _smooth_noise(rng, Pw, 37, prof[0], prof[1]) * 0.3 \
        + _smooth_noise(rng, Pw, 97, -2, 2) * 0.4
    far = far * s
    # the land only ever occupies the bottom rows: composite that band alone
    # (the whole-panorama version of this was 0.3 s of float work)
    base = (_smooth_noise(rng, Pw, 53, prof[2] * 0.6, prof[2] * 1.1)) * s
    wood = _smooth_noise(rng, Pw, 29, 0.0, 1.0)
    n_tr = int(Pw / 4)
    cx = rng.uniform(0, Pw, n_tr)
    rr_ = rng.uniform(2.5, 6.5, n_tr) * s
    hh_ = rng.uniform(prof[2], prof[3], n_tr) * s
    poplar = rng.uniform(0.0, 1.0, n_tr) < 0.22
    rr_ = np.where(poplar, rr_ * 0.45, rr_)                 # narrow ...
    hh_ = np.where(poplar, hh_ * 1.35, hh_)                 # ... and tall
    crowns = np.zeros(Pw)
    live = wood[cx.astype(int) % Pw] >= 0.42
    cx, rr_, hh_, poplar = cx[live], rr_[live], hh_[live], poplar[live]
    off = np.arange(-8, 9)[None, :]
    colf = np.floor(cx)[:, None] + off                    # (n, 17)
    dxc = (colf - cx[:, None]) / rr_[:, None]
    ax_ = np.clip(1 - np.abs(dxc), 0, 1)
    round_ = hh_[:, None] - rr_[:, None] + rr_[:, None] * np.sqrt(np.clip(1 - dxc ** 2, 0, 1))
    point_ = hh_[:, None] * ax_ ** 0.8
    bump = np.where(poplar[:, None], point_, round_)
    bump = np.where(np.abs(dxc) <= 1.0, bump, 0.0)
    np.maximum.at(crowns, (colf.astype(np.int64) % Pw).ravel(), bump.ravel())
    tree = np.maximum(base, crowns)
    # the foliage varies tree to tree: a fine noise on its brightness
    shade = _smooth_noise(rng, Pw, max(64, Pw // 7), -1.0, 1.0)[:, None, None]
    HB = int(min(R, max(far.max(), tree.max(), remote.max()) + 6))
    band = img[:, R - HB:].astype(np.float32)
    yy = y_up[None, R - HB:]                          # (1, HB)
    hz32 = np.array(HAZE_RGB, np.float32)
    a_rem = np.clip(remote[:, None] - yy + 0.5, 0.0, 1.0)[..., None] * 0.55
    c_rem = (np.array(HILL_FAR_BASE, np.float32) * 0.5 + hz32 * 0.5)
    band[:] = band * (1 - a_rem) + c_rem * a_rem
    a_far = np.clip(far[:, None] - yy + 0.5, 0.0, 1.0)[..., None]
    tf = np.clip(yy / np.maximum(far[:, None], 1.0), 0.0, 1.0)[..., None]
    fb, ft = np.array(HILL_FAR_BASE, np.float32), np.array(HILL_FAR, np.float32)
    band[:] = band * (1 - a_far) + (fb + (ft - fb) * tf) * a_far
    a_tr = np.clip(tree[:, None] - yy + 0.5, 0.0, 1.0)[..., None]
    tt = np.clip(yy / np.maximum(tree[:, None], 1.0), 0.0, 1.0)[..., None]
    tb, tt_ = np.array(TREE_LINE_BASE, np.float32), np.array(TREE_LINE, np.float32)
    c_tr = tb + (tt_ - tb) * tt ** 0.7 + shade * np.array([7.0, 9.0, 6.0], np.float32)
    band[:] = band * (1 - a_tr) + c_tr * a_tr
    # the land's foot takes the tone of the ground it stands on: its last
    # rows fade to LAND_FOOT_RGB (the ground at HAZE_LAND, which is where
    # draw_atmosphere caps it), so land and ground meet in one tone. It
    # used to fade 30 % toward the PALE haze while the ground under it went
    # to 88 %: a 6-8 px fog strip that floated the trees.
    foot = np.array(LAND_FOOT_RGB, np.float32)
    fade = (np.clip(1.0 - yy / 4.0, 0.0, 1.0)[..., None] * 0.85).astype(np.float32)
    band[:] = band + (foot - band) * fade
    img[:, R - HB:] = np.clip(band, 0, 255).astype(np.uint8)
    top = tuple(int(v) for v in img[:, 0].mean(axis=0))
    del img                                           # unlock the surface
    surf = sky_s
    try:
        surf = surf.convert()
    except pygame.error:
        pass
    # where the sky is clear and how high the land stands, per column: what
    # a check of the sky's gradient may sample (World.sky_columns)
    land_px = np.maximum(np.maximum(remote, far), tree)
    ent = dict(surf=surf, Pw=Pw, rows=R, top=top, cloud_edge=cloud_edge,
               clear=cloud_cov < 0.02, land_px=land_px,
               ms=(time.perf_counter() - t0) * 1e3)
    # ~11 MB an entry (5920 x 460 x 4): a handful covers every map at one
    # window size; a new size evicts the oldest, it does not grow forever
    while len(_PANO) >= PANO_CACHE:
        _PANO.pop(next(iter(_PANO)))
    _PANO[key] = ent
    return ent


# ======================================================================= #
#  STRIPE CLIPPING -- analytic band / convex-polygon intersection          #
# ======================================================================= #
def clip_bands(C: np.ndarray, m: np.ndarray, width: float, parity: int = 1) -> list:
    """The bands {k*w <= p.m < (k+1)*w, k odd (parity)} cut out of the convex
    polygon C (M,2), all at once: every edge contributes its start vertex
    when it is inside a band and its crossings of the band's two lines in the
    order they occur, which walks each band's polygon in order. One numpy
    pass for every band on screen; no per-band clipper call."""
    sv = C @ m
    k0, k1 = int(math.floor(sv.min() / width)), int(math.floor(sv.max() / width))
    ks = np.arange(k0, k1 + 1)
    ks = ks[np.mod(ks, 2) == parity]
    if not len(ks):
        return []
    a = ks * width
    b = a + width
    Vi, Vj = C, np.roll(C, -1, axis=0)
    si, sj = sv, np.roll(sv, -1)
    den = sj - si
    safe = np.where(np.abs(den) > 1e-12, den, 1e-12)
    inside = (si[None, :] >= a[:, None]) & (si[None, :] <= b[:, None])
    ta = (a[:, None] - si[None, :]) / safe
    tb = (b[:, None] - si[None, :]) / safe
    va, vb = (ta > 0) & (ta < 1), (tb > 0) & (tb < 1)
    up = (den > 0)[None, :]
    t1, v1 = np.where(up, ta, tb), np.where(up, va, vb)
    t2, v2 = np.where(up, tb, ta), np.where(up, vb, va)
    E = (Vj - Vi)[None, :, :]
    P0 = np.broadcast_to(Vi[None, :, :], (len(ks),) + Vi.shape)
    P1 = Vi[None] + t1[..., None] * E
    P2 = Vi[None] + t2[..., None] * E
    pts = np.stack([P0, P1, P2], axis=2).reshape(len(ks), -1, 2)
    msk = np.stack([inside, v1, v2], axis=2).reshape(len(ks), -1)
    return [pts[k][msk[k]] for k in range(len(ks)) if msk[k].sum() >= 3]


# ======================================================================= #
#  THE WORLD                                                               #
# ======================================================================= #
class World:
    """Built once per Renderer (per track). Every method takes the renderer."""

    def __init__(self, rnd):
        tr = rnd.track
        self.track = tr
        self.lay = scn.layout(tr)
        self.theme = self.lay.theme
        # refreshed from rnd.cfg every frame (draw_backdrop): a live setting
        self.detail = getattr(rnd.cfg, 'detail', 'high')
        n = (len(tr.s) - 1) if tr.closed else len(tr.s)
        self._n = n
        self._nrm = np.column_stack([-np.sin(tr.psi), np.cos(tr.psi)])
        self._grid = None               # the surface raster for skid colours
        self._haze_key = None
        self._haze_surf = None
        self._haze_y0 = 0
        self._pano = None
        self._pano_lens = None          # the panorama rescaled to another lens
        self._chunk_poly = [ch['poly'] for ch in self.lay.chunks]
        self._chunk_kind = [ch['kind'] for ch in self.lay.chunks]
        # standing water: streaks of reflected sky, seeded, per patch / area
        self._water = self._water_layout(tr)
        # built NOW, not on the first chase frame: the panorama is ~50 ms and
        # the decimation masks a few, and a first-frame hitch is what a frame
        # budget's p99 (and a driver pressing C) would see. Cached per
        # process, so a second Renderer of the same map pays nothing.
        keep_masks(tr)
        c3 = getattr(rnd, '_chase', None)
        if c3 is not None:
            p = build_panorama(self.lay.name, self.theme, c3.fl)
            self._pano = dict(p, fl=float(c3.fl))
        # the open map's pad: verge ring, slab joints
        self._pad = None
        for a in getattr(tr, 'areas', []):
            if a.drivable and a.kind == 'rrect':
                cx, cy, hx, hy, r = a.params
                gv = scn.VERGE_W
                ring = type(a)('rrect', (cx, cy, hx + gv, hy + gv, r + gv)).polygon(12)
                # the verge ring as an ANNULUS -- outer outline, then the
                # pad's own outline backwards: the even-odd fill paints only
                # the 1.4 m band, not a second screen-sized polygon under
                # the pad (0.4 ms each in a plan view)
                inner = a.polygon(12)
                annulus = np.vstack([ring, ring[:1], inner[:1], inner[::-1], inner[-1:]])
                self._pad = dict(area=a, ring=ring, annulus=annulus)
        # the surface raster (skid colours) and the tufts it places: 1-6 ms,
        # built now so the first skid mark or chase frame is not a hitch --
        # whatever the detail level, which can be raised mid-session
        self._surface_grid()
        self._tuft_layout()

    # ------------------------------------------------------------------ #
    def _water_layout(self, tr):
        """Puddles -- irregular 8-gons of water that catch more sky than the
        wet film round them -- for every wet SurfacePatch and wet Area.
        World metres, seeded; drawn only where seen. (Thin streaks along the
        road were tried first: in the chase view they converge on the
        vanishing point and read as lane markings.)"""
        rng = scn.rng_for(tr, 'water')
        nrm = self._nrm
        hw = 0.5 * tr.width
        tang = np.column_stack([np.cos(tr.psi), np.sin(tr.psi)])
        ang = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)

        def puddle(c, u, v, a_, b_):
            # a squashed, lumpy ellipse: radius jittered per vertex
            rj = rng.uniform(0.75, 1.15, 8)
            return (c[None, :] + (a_ * rj * np.cos(ang))[:, None] * u[None, :]
                    + (b_ * rj * np.sin(ang))[:, None] * v[None, :])
        out = []
        for p in tr.surfaces:
            if p.mu_scale >= 0.95:
                continue
            L = tr.length
            a, b = p.s0, (p.s1 if p.s1 >= p.s0 else p.s1 + L)
            n0, n1 = max(p.n0, -hw), min(p.n1, hw)
            k = int(max(4, (b - a) * (n1 - n0) / 48.0))
            for _ in range(k):
                a_ = float(rng.uniform(0.6, 2.4))
                b_ = a_ * float(rng.uniform(0.25, 0.6))
                sc = float(rng.uniform(a + a_, b - a_))
                nc = float(rng.uniform(n0 + b_, n1 - b_))
                i = int(scn._i_of_s(tr, sc))
                c = tr.xy[i] + nc * nrm[i]
                out.append((sc % L if tr.closed else sc,
                            puddle(c, tang[i], nrm[i], a_, b_), int(rng.integers(0, 2))))
        quads = np.array([q for _s, q, _c in out]).reshape(-1, 8, 2)
        ss = np.array([s_ for s_, _q, _c in out])
        cc = np.array([c for _s, _q, c in out], dtype=np.int8)
        # wet AREAS (the open map's square) in world coordinates
        aq = []
        for ar in getattr(tr, 'areas', []):
            if ar.mu_scale >= 0.95 or ar.kind != 'rect':
                continue
            x0, y0, x1, y1 = ar.params
            k = int((x1 - x0) * (y1 - y0) / 80.0)
            for _ in range(k):
                a_ = float(rng.uniform(0.6, 2.4))
                b_ = a_ * float(rng.uniform(0.3, 0.7))
                c = np.array([float(rng.uniform(x0 + a_, x1 - a_)),
                              float(rng.uniform(y0 + a_, y1 - a_))])
                t_ = float(rng.uniform(0.0, math.pi))
                u = np.array([math.cos(t_), math.sin(t_)])
                aq.append(puddle(c, u, np.array([-u[1], u[0]]), a_, b_))
        return dict(quads=quads, s=ss, col=cc,
                    area_quads=np.array(aq, dtype=np.float64).reshape(-1, 8, 2))

    # ------------------------------------------------------------------ #
    #  BACKDROP                                                          #
    # ------------------------------------------------------------------ #
    def draw_backdrop(self, rnd) -> bool:
        """Sky + ground fill. True = drawn; False = the renderer's own
        background (render._draw_sky3 / C_BG) is used."""
        sc = rnd.screen
        c3 = rnd._cam3
        W, H = rnd.W, rnd.H
        # the detail level is a live setting (drive.Sim's options menu flips
        # cfg.detail in place): read once a frame, here, the first world
        # layer of every frame
        self.detail = getattr(rnd.cfg, 'detail', 'high')
        if c3 is None:
            sc.fill(GRASS_RGB)
            return True
        iy = int(round(c3.horizon_y))
        if iy < H:
            y0 = max(iy, 0)
            sc.fill(GRASS_RGB, (0, y0, W, H - y0))
        if iy <= 0:
            return True
        pano = self._pano
        if pano is None:                 # (no chase camera at construction)
            p = build_panorama(self.lay.name, self.theme, c3.fl)
            pano = self._pano = dict(p, fl=float(c3.fl))
        # a lens other than the one it was built for: a rescaled copy
        pano = self._pano_for_lens(float(c3.fl))
        Pw, R = pano['Pw'], pano['rows']
        # the scroll follows the view's ACTUAL heading: the chase camera
        # leans into corners, so its forward axis is a few degrees off the
        # lagged psi_cam, and 3 deg of that is 50 px of sky sliding against
        # the ground and the props
        az = math.atan2(float(c3._f[1]), float(c3._f[0]))
        cc = (-az * pano['fl']) % Pw
        top = iy - R
        if top > 0:
            sc.fill(pano['top'], (0, 0, W, top))
        sy0 = max(0, -top)
        hgt = min(R - sy0, H - max(top, 0))
        if hgt <= 0:
            return True
        x0 = int(round(cc - 0.5 * W)) % Pw
        w1 = min(W, Pw - x0)
        surf = pano['surf']
        dy = max(top, 0)
        sc.blit(surf, (0, dy), (x0, sy0, w1, hgt))
        if w1 < W:
            sc.blit(surf, (w1, dy), (0, sy0, W - w1, hgt))
        return True

    def sky_columns(self, rnd, above_px: float) -> np.ndarray:
        """Screen columns where the panorama is clear sky -- no cloud, and
        the land lower than `above_px` px above the horizon: the columns a
        check of the sky's gradient may honestly sample. Chase only."""
        c3, pano = rnd._cam3, self._pano
        if c3 is None or pano is None:
            return np.zeros(0, dtype=np.int64)
        Pw = pano['Pw']
        az = math.atan2(float(c3._f[1]), float(c3._f[0]))
        cc = (-az * pano['fl']) % Pw
        x = np.arange(rnd.W)
        col = np.rint(cc + (x - 0.5 * rnd.W) * pano['fl'] / float(c3.fl)).astype(np.int64) % Pw
        ok = pano['clear'][col] & (pano['land_px'][col] * float(c3.fl) / pano['fl'] < above_px)
        return x[ok]

    def _pano_for_lens(self, fl: float) -> dict:
        """The panorama at focal length `fl`: the one built with the World
        when fl is within PANO_FL_TOL of it, else ONE rescaled copy of it
        (pygame.transform.scale, 1.8 ms at 1280x800), kept until the lens
        moves PANO_FL_TOL away again. Exact: columns are fl * azimuth, rows
        fl * tan(elevation), so a lens change is a uniform scale. Never a
        rebuild (60 ms) inside a frame; a speed-widened lens rescales ~15
        times over a 0-45 m/s run, and not at all at a steady speed (the
        per-frame scale of the visible window it replaces was 0.55 ms)."""
        base = self._pano
        if abs(fl / base['fl'] - 1.0) <= PANO_FL_TOL:
            return base
        cur = self._pano_lens
        if cur is not None and abs(fl / cur['fl'] - 1.0) <= PANO_FL_TOL:
            return cur
        k = fl / base['fl']
        Pw = max(int(round(base['Pw'] * k)), 16)
        rows = max(int(round(base['rows'] * k)), 1)
        surf = pygame.transform.scale(base['surf'], (Pw, rows))
        self._pano_lens = dict(base, surf=surf, Pw=Pw, rows=rows,
                               fl=base['fl'] * Pw / base['Pw'])
        return self._pano_lens

    # ------------------------------------------------------------------ #
    #  GROUND -- mowing stripes (the speed cue the 20 m grid used to be)  #
    # ------------------------------------------------------------------ #
    def _ground_rays(self, c3, sx, sy):
        """Screen points -> the ground points under them (chase)."""
        f, r, u = c3._f, c3._r, c3._u
        W, H, fl = c3.W, c3.H, c3.fl
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        dx = f[0] + ((sx - 0.5 * W) / fl) * r[0] - ((sy - 0.5 * H) / fl) * u[0]
        dy = f[1] + ((sx - 0.5 * W) / fl) * r[1] - ((sy - 0.5 * H) / fl) * u[1]
        dz = f[2] + ((sx - 0.5 * W) / fl) * r[2] - ((sy - 0.5 * H) / fl) * u[2]
        t = c3.eye[2] / np.maximum(-dz, 1e-9)
        return np.column_stack([c3.eye[0] + t * dx, c3.eye[1] + t * dy])

    def _row_of_depth(self, c3, d):
        """Screen row where the ground is at camera depth d (chase)."""
        f, u = c3._f, c3._u
        return 0.5 * c3.H + c3.fl * (f[2] + c3.eye[2] / d) / max(u[2], 1e-9)

    def view_quads(self, rnd) -> list:
        """The visible ground as convex world quads: [(quad, zone)]. Chase:
        the frustum's footprint between the bottom of the frame and
        STRIPE_NEAR_D (zone 0), then out to STRIPE_FAR_D (zone 1). Plan: the
        screen rectangle mapped back to the world (zone 0)."""
        W, H = rnd.W, rnd.H
        c3 = rnd._cam3
        if c3 is None:
            S = np.array([[-8, -8], [W + 8, -8], [W + 8, H + 8], [-8, H + 8]], dtype=np.float64)
            Rm = rnd._Rm
            P = ((S - rnd._anchor) / np.array([rnd.ppm, -rnd.ppm])) @ Rm + rnd.cam
            return [(P, 0)]
        out = []
        y_bot = H + 4.0
        xl, xr = -0.12 * W, 1.12 * W
        prev = y_bot
        for zone, d in ((0, STRIPE_NEAR_D), (1, STRIPE_FAR_D)):
            yf = self._row_of_depth(c3, d)
            if yf >= prev - 1.0:
                continue
            yf = max(yf, c3.horizon_y + 1.0)
            P = self._ground_rays(c3, [xl, xr, xr, xl], [yf, yf, prev, prev])
            out.append((P, zone))
            prev = yf
        return out

    def draw_ground(self, rnd) -> None:
        """Grass, verge, run-off and gravel: under the ribbon. Here: the
        world-anchored mowing stripes (the verge is drawn with the ribbon,
        the traps and run-off after it, in draw_surface)."""
        sc = rnd.screen
        lay = self.lay
        m = np.array([-lay.stripe_dir[1], lay.stripe_dir[0]])
        polys, cols = [], []
        mid = tuple((a + b) // 2 for a, b in zip(GRASS_RGB, GRASS2_RGB))
        for C, zone in self.view_quads(rnd):
            if zone == 1 and self.detail == 'low':
                continue
            for P in clip_bands(C, m, scn.STRIPE_W, 1):
                polys.append(P)
                cols.append(GRASS2_RGB if zone == 0 else mid)
        for k, pts in project_polys(rnd, polys):
            pygame.draw.polygon(sc, cols[k], pts)
        # the tufts are detail -- except the zoomed-out plan view's sparse
        # ones, which are its speed cue: ~0.2 ms at 7-9 px/m (see _tufts)
        if self.detail != 'low' or (rnd._cam3 is None and rnd.ppm < TUFT_PPM):
            self._tufts(rnd)

    def _tuft_layout(self):
        """Clumps of darker, lusher and drier grass on a jittered 4.5 m grid,
        kept only where the raster says grass: texture for the near field of
        the chase view and a speed cue that streams past, as a real verge's
        weeds do. World-anchored, seeded from the track name."""
        if getattr(self, '_tq', None) is not None:
            return self._tq
        g, gx0, gy0, cell = self._surface_grid()
        rng = scn.rng_for(self.track, 'tufts')
        x1, y1 = gx0 + g.shape[0] * cell, gy0 + g.shape[1] * cell
        xs = np.arange(gx0 + 0.5 * TUFT_PITCH, x1, TUFT_PITCH)
        ys = np.arange(gy0 + 0.5 * TUFT_PITCH, y1, TUFT_PITCH)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        C = np.column_stack([X.ravel(), Y.ravel()])
        C += rng.uniform(-0.45, 0.45, C.shape) * TUFT_PITCH
        a = rng.uniform(0.3, 0.8, len(C))            # half-length, m
        b = a * rng.uniform(0.35, 0.7, len(C))       # half-width
        th = rng.uniform(0.0, math.pi, len(C))
        col = rng.integers(0, 3, len(C))
        # a lumpy hexagon; every vertex must be on grass (a tuft never
        # overhangs the verge)
        u = np.column_stack([np.cos(th), np.sin(th)])
        v = np.column_stack([-u[:, 1], u[:, 0]])
        ang = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
        rj = rng.uniform(0.7, 1.15, (len(C), 6))
        Q = (C[:, None, :] + (a[:, None] * rj * np.cos(ang))[..., None] * u[:, None, :]
             + (b[:, None] * rj * np.sin(ang))[..., None] * v[:, None, :])
        ix = np.clip(((Q[..., 0] - gx0) / cell).astype(int), 0, g.shape[0] - 1)
        iy = np.clip(((Q[..., 1] - gy0) / cell).astype(int), 0, g.shape[1] - 1)
        ok = (g[ix, iy] == 0).all(axis=1)
        # the plan view's sparse cue: one clump in TUFT_SPARSE, a fixed
        # (seeded) subset so the same clumps stream past every lap
        sparse = (rng.integers(0, TUFT_SPARSE, len(C)) == 0) & ok
        # ... drawn 1.6x larger, the dry ones light and the rest dark
        sC = C[sparse]
        sQ = sC[:, None, :] + 1.6 * (Q[sparse] - sC[:, None, :])
        self._tq = dict(Q=Q[ok], C=C[ok], col=col[ok], sC=sC, sQ=sQ,
                        scol=(col[sparse] == 2).astype(np.int64))
        return self._tq

    def _tufts(self, rnd) -> None:
        """Chase: the tufts from TUFT_NEAR to TUFT_DEPTH, their colour
        fading from the grass's own to full contrast by TUFT_FULL. Plan: all
        of them when zoomed in (>= TUFT_PPM); zoomed out -- at speed -- one
        in TUFT_SPARSE, drawn 1.6x larger and darker / drier, so the grass
        still visibly streams past when the car runs along the mowing
        stripes (the reviewer measured 0 % of the grass changing frame to
        frame at 40 m/s: the stripes are the only other ground detail)."""
        c3 = rnd._cam3
        tq = self._tuft_layout()
        C = tq['C']
        if not len(C):
            return
        sc = rnd.screen
        if c3 is not None:
            d = c3.ground_depth(C)
            m = (d > TUFT_NEAR) & (d < TUFT_DEPTH)
            # and inside the frame's width at that depth (+ a margin)
            lat = ((C[:, 0] - c3.eye[0]) * c3._r[0] + (C[:, 1] - c3.eye[1]) * c3._r[1])
            m &= np.abs(lat) < (0.62 * rnd.W / c3.fl) * np.maximum(d, 1.0) + 2.0
            sel = np.flatnonzero(m)
            if not len(sel):
                return
            w = np.clip((d[sel] - TUFT_NEAR) / (TUFT_FULL - TUFT_NEAR), 0.0, 1.0)[:, None]
            mid = np.array(GRASS_RGB, np.float64) * 0.5 + np.array(GRASS2_RGB, np.float64) * 0.5
            cols = np.rint(mid + (_TUFT_ARR[tq['col'][sel]] - mid) * w).astype(np.int64).tolist()
            for k, pts in project_polys(rnd, tq['Q'][sel]):
                pygame.draw.polygon(sc, cols[k], pts)
            return
        # plan: no clipping to do, so no project_polys -- cull the centres
        # to the disc, then to the SCREEN (one small transform), and project
        # only the survivors' outlines. Each outline reaches Python as its
        # OWN short list, freed before the next is made: P.tolist() of them
        # all held ~130 x 7 lists alive at once, past the collector's gen-0
        # threshold (700) EVERY frame -- arena car_up at 30 m/s ran 274
        # gen-0 and 2 gen-2 collections per 300 frames (11 and 0 with the
        # tufts off, and now), and the 4-5 ms gen-2 pauses landed here. A
        # clump costs ~1.1 us (0.6 turning its 12 numbers into Python ints,
        # 0.45 the fill) plus the culls: the sparse layer is ~0.2 ms for the
        # 120-145 clumps a 1280x800 frame shows at 7-9 px/m, ~0.7 ms for the
        # ~790 at the widest manual zoom (0.35, 3 px/m).
        if rnd.ppm >= TUFT_PPM:
            C, Q, col, pal = tq['C'], tq['Q'], tq['col'], TUFT_RGB
        else:
            C, Q, col, pal = tq['sC'], tq['sQ'], tq['scol'], TUFT_PLAN_RGB
        r = rnd._view_radius() + 2.0
        sel = np.flatnonzero((C[:, 0] - rnd.cam[0]) ** 2 + (C[:, 1] - rnd.cam[1]) ** 2 < r * r)
        if not len(sel):
            return
        S = rnd.world_to_screen(C[sel])
        pad = 2.5 * rnd.ppm + 2.0                  # a clump is < 2.5 m across
        on = ((S[:, 0] > -pad) & (S[:, 0] < rnd.W + pad)
              & (S[:, 1] > -pad) & (S[:, 1] < rnd.H + pad))
        sel = sel[on]
        if not len(sel):
            return
        P = rnd.world_to_screen(Q[sel].reshape(-1, 2)).astype(np.int32).reshape(len(sel), -1, 2)
        for pts, c in zip(P, col[sel].tolist()):
            pygame.draw.polygon(sc, pal[c], pts.tolist())

    # ------------------------------------------------------------------ #
    #  THE RIBBON AND THE VERGE (render._draw_ribbon hands over to this)  #
    # ------------------------------------------------------------------ #
    def ribbon(self, rnd, runs) -> None:
        """The verge (a band VERGE_W wide outside each edge, as ONE wider
        polygon under the ribbon) and then THE ribbon, one polygon per run
        exactly as render has always drawn it."""
        tr = self.track
        C_TARMAC = _R().C_TARMAC
        hw = 0.5 * tr.width
        polys, cols = [], []
        pieces = [jj for idx in runs for jj in self._depth_pieces(rnd, idx)]
        if self.lay.verge:
            # every verge first, then every ribbon: a run's verge must not
            # paint over another run's tarmac where two stretches pass close
            g = hw + scn.VERGE_W
            for jj in pieces:
                P, nr = tr.xy[jj], self._nrm[jj]
                polys.append(np.vstack([P + g * nr, (P - g * nr)[::-1]]))
                cols.append(VERGE_RGB)
        for jj in pieces:
            polys.append(np.vstack([tr.left[jj], tr.right[jj][::-1]]))
            cols.append(C_TARMAC)
        for k, pts in project_polys(rnd, polys):
            pygame.draw.polygon(rnd.screen, cols[k], pts)

    def _depth_pieces(self, rnd, idx) -> list:
        """A run's samples cut into consecutive pieces at the camera depths
        STRIP_SPLIT_D -- chase only. One polygon from the bumper to the
        horizon pays twice: its fill tests every edge on every row it spans
        (the 120 far vertices on all 470 rows), and the one vertex near the
        eye sends ALL of it through the clipper. Cut by depth, the far
        pieces are many vertices on a few rows and only the short near piece
        is clipped: the arena ribbon at T3 measured 0.23 + 0.23 ms whole,
        0.19 + 0.06 ms in pieces.

        Each piece runs ONE SAMPLE PAST the cut, so neighbours overlap by a
        segment instead of abutting on a shared edge: pygame's fill of two
        polygons meeting on a slanted edge leaves pixels along it that
        neither owns, and the verge green drawn under the ribbon showed
        through as a dotted line across the tarmac ~8 m ahead (the skidpad
        at n = 3.5 m, 20 m/s: 37-46 of 120 frames, up to 206 px; a pose
        every 3.7 m round it: 44 of 168 frames; 0 of either now). The
        pieces are one colour, so the overlap is invisible; it costs one
        vertex per cut."""
        c3 = rnd._cam3
        if len(idx) < 2:
            return []                   # a strip needs two samples
        if c3 is None or len(idx) < 8:
            return [idx]
        zone = np.digitize(c3.ground_depth(self.track.xy[idx]), STRIP_SPLIT_D)
        cut = np.flatnonzero(np.diff(zone)) + 1
        if not len(cut):
            return [idx]
        keep, last = [], 0
        for c in cut.tolist():
            if c - last >= 3 and len(idx) - c >= 2:     # no sliver pieces
                keep.append(c)
                last = c
        b = [0] + keep + [len(idx)]
        return [idx[a:min(e + 2, len(idx))] for a, e in zip(b[:-1], b[1:])]

    # ------------------------------------------------------------------ #
    #  SURFACE -- traps, run-off, rubber, repairs, cracks                #
    # ------------------------------------------------------------------ #
    def _near_chunks(self, rnd):
        lay = self.lay
        if not len(lay.chunk_c):
            return np.zeros(0, dtype=np.int64)
        r = rnd._view_radius() * 1.3
        d = np.hypot(lay.chunk_c[:, 0] - rnd.cam[0], lay.chunk_c[:, 1] - rnd.cam[1])
        m = d < r + lay.chunk_r
        if rnd._cam3 is not None:
            m &= rnd._cam3.ground_depth(lay.chunk_c) > -lay.chunk_r
        return np.flatnonzero(m)

    def draw_surface(self, rnd, runs, windows) -> None:
        """Detail on the tarmac, after the ribbon and before the patches --
        and the traps / run-off beside it, which adjoin the edge and so are
        drawn after the verge."""
        sc = rnd.screen
        tr = self.track
        lay = self.lay
        low = self.detail == 'low'
        # --- gravel traps and painted run-off
        sel = self._near_chunks(rnd)
        if len(sel):
            polys = [self._chunk_poly[i] for i in sel]
            for k, pts in project_polys(rnd, polys):
                kind = self._chunk_kind[sel[k]]
                pygame.draw.polygon(sc, GRAVEL_RGB if kind == 'gravel' else RUNOFF_RGB, pts)
            if len(lay.runoff_paint):
                c = lay.runoff_paint.mean(axis=1)
                m = self._cull_pts(rnd, c, 6.0)
                for k, pts in project_polys(rnd, lay.runoff_paint[m]):
                    pygame.draw.polygon(sc, RUNOFF_PAINT, pts)
            if len(getattr(lay, 'trap_edges', ())):
                self._chains(rnd, lay.trap_edges, GRAVEL_EDGE, 0.12)
            if len(lay.rakes) and not low:
                self._chains(rnd, lay.rakes, GRAVEL_RAKE, 0.0)
        # --- the rubbered racing line (a strip per run, n varying along it)
        #     and the drag groove, through the painted-line path so their far
        #     ends stay continuous
        items = []
        if lay.n_rl is not None:
            items += [(idx, lay.n_rl[idx], 2.0 * scn.RL_HALF, RUBBER_RGB)
                      for idx in runs if len(idx) >= 2]
        grooves = getattr(lay, 'grooves', [])
        if grooves and not tr.closed:
            for idx in runs:
                if len(idx) < 2:
                    continue
                i_lo, i_hi = int(idx[0]), int(idx[-1])
                for s0, s1, ny, hw_, shade in grooves:
                    # each band from ITS OWN boundary samples, not from the
                    # first / last kept sample of the decimated run inside
                    # it: those are up to 30 m apart on a straight, which
                    # left 8-30 m of bare strip between two bands (s ~
                    # 190-205 in a plan view). An open track's run is sorted.
                    b0, b1 = max(int(round(s0 / tr.ds)), i_lo), min(int(round(s1 / tr.ds)), i_hi)
                    if b1 - b0 < 1:
                        continue
                    a_, e_ = np.searchsorted(idx, [b0, b1], side='left')
                    mid = idx[a_ + int(idx[a_] == b0):e_]
                    ii = np.concatenate([[b0], mid, [b1]]) if (len(mid) == 0 or mid[-1] != b1) \
                        else np.concatenate([[b0], mid])
                    items.append((ii, ny, 2.0 * hw_, GROOVE_RGB[shade]))
        self._paint_lines(rnd, items)
        # --- asphalt repairs and sealed cracks
        L = tr.length
        if len(lay.patches):
            c = lay.patches.mean(axis=1)
            m = self._cull_pts(rnd, c, 8.0)
            for k, pts in project_polys(rnd, lay.patches[m]):
                col = PATCH_RGB[int(lay.patch_col[np.flatnonzero(m)[k]])]
                pygame.draw.polygon(sc, col, pts)
                if not low:
                    pygame.draw.polygon(sc, PATCH_SEAL, pts, 1)
        if len(lay.cracks) and not low:
            m = _in_windows(lay.crack_s, windows, L, tr.closed, pad=2.0)
            con = None
            if rnd._cam3 is not None and m.any():
                # chase: CRACK_NEAR..CRACK_FAR only, the contrast ramping in
                # and out at both ends. Nearer, a 1-2 px line 300 px long
                # read as a black scratch across the lane; deeper, a 3 cm
                # crack is a third of a pixel
                dd = rnd._cam3.ground_depth(lay.cracks[m].mean(axis=1))
                keep = (dd > CRACK_NEAR) & (dd < CRACK_FAR)
                m[m] = keep
                dd = dd[keep]
                con = (np.clip((dd - CRACK_NEAR) / 8.0, 0.0, 1.0)
                       * np.clip((CRACK_FAR - dd) / 15.0, 0.0, 1.0))
            if m.any():
                A, B, live = project_segs(rnd, lay.cracks[m, 0], lay.cracks[m, 1])
                # sealed cracks are ~4 cm of tar: 3 px at 12 m, 1 px beyond 40
                wpx = np.clip(np.round(0.03 * _depth_px(
                    rnd, 0.5 * (lay.cracks[m, 0] + lay.cracks[m, 1]))), 1, 2).astype(int)
                if con is None:
                    cols = [CRACK_RGB] * len(wpx)
                else:
                    t0_ = np.array(_R().C_TARMAC, np.float64)
                    cols = np.rint(t0_ + (np.array(CRACK_RGB, np.float64) - t0_)
                                   * con[:, None]).astype(np.int64).tolist()
                for k in np.flatnonzero(live).tolist():
                    pygame.draw.line(sc, cols[k], A[k], B[k], int(wpx[k]))

    def _chains(self, rnd, E, col, w_m) -> None:
        """Segments (K,2,2) that mostly join end to start (trap borders, rake
        lines) as polylines: ONE draw call per chain, split where a segment
        does not continue the last or where the width (w_m at its depth,
        1 px minimum) changes. Per-segment draw calls here were ~0.3 ms."""
        m = self._cull_pts(rnd, E.mean(axis=1), 2.0, max_depth=DETAIL_DEPTH)
        if not m.any():
            return
        Em = E[m]
        A, B, live = project_segs(rnd, Em[:, 0], Em[:, 1])
        if w_m > 0.0:
            wpx = np.clip(np.round(w_m * _depth_px(rnd, Em.mean(axis=1))), 1, 3).astype(int)
        else:
            wpx = np.ones(len(Em), dtype=int)
        cont = np.r_[False, (np.abs(Em[1:, 0] - Em[:-1, 1]).sum(axis=1) < 1e-6)
                     & live[1:] & live[:-1] & (wpx[1:] == wpx[:-1])]
        sc = rnd.screen
        Al, Bl, wl = A.tolist(), B.tolist(), wpx.tolist()
        pts, w_ = None, 1
        for k in range(len(Em)):
            if not live[k]:
                continue
            if cont[k] and pts is not None:
                pts.append(Bl[k])
                continue
            if pts is not None and len(pts) >= 2:
                pygame.draw.lines(sc, col, False, pts, w_)
            pts, w_ = [Al[k], Bl[k]], wl[k]
        if pts is not None and len(pts) >= 2:
            pygame.draw.lines(sc, col, False, pts, w_)

    def _cull_pts(self, rnd, C, rad, max_depth=None) -> np.ndarray:
        """World points within the view (and, chase, not behind the eye --
        nor, for fine detail, deeper than `max_depth`, where it would be a
        sub-pixel smudge in the haze and cost as much as a near one)."""
        r = rnd._view_radius() * 1.3 + rad
        m = (C[:, 0] - rnd.cam[0]) ** 2 + (C[:, 1] - rnd.cam[1]) ** 2 < r * r
        if rnd._cam3 is not None and m.any():
            d = rnd._cam3.ground_depth(C)
            m &= d > -rad
            if max_depth is not None:
                m &= d < max_depth + rad
        return m

    # ------------------------------------------------------------------ #
    #  STANDING WATER (render._draw_patches hands over to this)           #
    # ------------------------------------------------------------------ #
    def patches(self, rnd, windows) -> None:
        """Wet / damp patches: a damp rim, the patch in its own colour (the
        one render has always drawn, which the self-checks count), and
        streaks of reflected sky -- brighter far away, where water reflects
        at a grazing angle, which is what makes it read as WATER."""
        tr = self.track
        if not tr.surfaces:
            return
        sc = rnd.screen
        L = tr.length
        hw = 0.5 * tr.width
        polys, cols = [], []
        for s0, s1 in windows:
            for p in tr.surfaces:
                pa, pb = p.s0, p.s1
                if tr.closed and pb < pa:
                    pb += L
                hit = rnd._overlaps(pa, pb, s0, s1)
                if hit is None:
                    continue
                a, b = hit
                step = max(tr.ds, (b - a) / 40.0)
                ss = np.arange(a, b + 0.5 * step, step)
                ii = scn._i_of_s(tr, ss)
                n0, n1 = max(p.n0, -hw), min(p.n1, hw)
                wet = p.mu_scale < 0.95
                if wet:
                    # the rim: 0.5 m wider, 2 m longer, damp not wet -- as
                    # its two end caps and (where the patch stops short of
                    # an edge) its side bands, not one polygon under the
                    # whole patch: that overdrew the lower half of the
                    # frame a second time in the near field (0.36 ms of
                    # fill for the 14 patch polygons at T3)
                    aa, bb = max(a - 2.0, s0), min(b + 2.0, s1)
                    m0, m1 = max(n0 - 0.5, -hw), min(n1 + 0.5, hw)
                    caps = [(aa, a), (b, bb)]
                    caps = [(c0, c1) for c0, c1 in caps if c1 - c0 > 0.05]
                    if caps:
                        for q in scn.quads_at(tr, [c[0] for c in caps], [c[1] for c in caps],
                                              m0, m1, self._nrm):
                            polys.append(q)
                            cols.append(WATER_FRINGE)
                    for e0, e1 in ((m0, n0), (n1, m1)):
                        if e1 - e0 > 0.05:
                            polys.append(np.vstack([tr.xy[ii] + e0 * self._nrm[ii],
                                                    (tr.xy[ii] + e1 * self._nrm[ii])[::-1]]))
                            cols.append(WATER_FRINGE)
                zone = np.zeros(len(ii), dtype=np.int64)
                if wet and rnd._cam3 is not None:
                    d = rnd._cam3.ground_depth(tr.xy[ii])
                    zone = np.digitize(d, (25.0, 45.0, 90.0))
                cut = np.flatnonzero(np.diff(zone)) + 1
                for a_, b_ in zip(np.concatenate([[0], cut]),
                                  np.concatenate([cut, [len(zone)]])):
                    b2 = min(b_ + 1, len(zone))
                    if b2 - a_ < 2:
                        continue
                    jj = ii[a_:b2]
                    polys.append(np.vstack([tr.xy[jj] + n0 * self._nrm[jj],
                                            (tr.xy[jj] + n1 * self._nrm[jj])[::-1]]))
                    z = int(zone[a_])
                    cols.append(tuple(p.colour) if z == 0 else WATER_SHEEN[z - 1])
        for k, pts in project_polys(rnd, polys):
            pygame.draw.polygon(sc, cols[k], pts)
        wq = self._water['quads']
        if len(wq) and self.detail != 'low':
            m = _in_windows(self._water['s'], windows, L, tr.closed, pad=3.0)
            self._streaks(rnd, wq[m])

    def _streaks(self, rnd, Q) -> None:
        if not len(Q):
            return
        c3 = rnd._cam3
        if c3 is not None:
            d = c3.ground_depth(Q.mean(axis=1))
        else:
            d = np.zeros(len(Q))
        for k, pts in project_polys(rnd, Q):
            pygame.draw.polygon(rnd.screen, WATER_STREAK2 if d[k] > 40.0 else WATER_STREAK, pts)

    # ------------------------------------------------------------------ #
    #  KERBS (render._draw_kerbs hands over to this)                      #
    # ------------------------------------------------------------------ #
    def _kerb_geometry(self) -> dict:
        """Every kerb block's top, its two long faces and its colours, built
        ONCE (the layout is static): per frame the kerbs only select the
        blocks in the windows. Rebuilding these arrays for ~190 blocks every
        frame was ~0.1 ms of the 0.4 ms kerb layer."""
        kg = getattr(self, '_kg', None)
        if kg is not None:
            return kg
        R_ = _R()
        tr, lay = self.track, self.lay
        hw = 0.5 * tr.width
        i0 = scn._i_of_s(tr, lay.kerb_s0)
        i1 = scn._i_of_s(tr, lay.kerb_s1)
        sd = lay.kerb_side.astype(np.float64)[:, None]
        n_in, n_out = sd * (hw - scn.KERB_W), sd * hw
        P0, P1 = tr.xy[i0], tr.xy[i1]
        N0, N1 = self._nrm[i0], self._nrm[i1]
        a_in, a_out = P0 + n_in * N0, P0 + n_out * N0
        b_in, b_out = P1 + n_in * N1, P1 + n_out * N1
        K = len(i0)
        z0, zh = np.zeros((K, 1)), np.full((K, 1), KERB_H)

        def p3(P, z):
            return np.hstack([P, z])
        faces = []
        for A_, B_, sgn in ((a_in, b_in, -1.0), (a_out, b_out, 1.0)):
            faces.append(dict(q=np.stack([p3(A_, z0), p3(B_, z0), p3(B_, zh), p3(A_, zh)], axis=1),
                              mid=0.5 * (A_ + B_),
                              nrm=sgn * sd * 0.5 * (N0 + N1)))     # outward, horizontal
        cols = [R_.C_KERB_A if c == 0 else R_.C_KERB_B for c in lay.kerb_col.tolist()]
        self._kg = dict(
            flat=np.stack([a_in, a_out, b_out, b_in], axis=1),
            tops=np.stack([p3(a_in, zh), p3(a_out, zh), p3(b_out, zh), p3(b_in, zh)], axis=1),
            faces=faces, mid=0.5 * (P0 + P1), cols=cols,
            fcols=[tuple(int(v * KERB_FACE_K) for v in c) for c in cols])
        return self._kg

    def kerbs(self, rnd, windows) -> None:
        """Inside, apex and exit kerbs from the layout, all visible blocks in
        ONE projection. Chase: raised KERB_H with the face that looks at the
        eye drawn shaded under the top -- the kerb reads as a solid thing."""
        lay = self.lay
        if not len(lay.kerb_s0):
            return
        tr = self.track
        L = tr.length
        m = np.zeros(len(lay.kerb_s0), dtype=bool)
        shifts = (-L, 0.0, L) if tr.closed else (0.0,)
        for w0, w1 in windows:
            for k in shifts:
                m |= (lay.kerb_s1 + k > w0) & (lay.kerb_s0 + k < w1)
        sel = np.flatnonzero(m)
        if not len(sel):
            return
        kg = self._kerb_geometry()
        cols_all = kg['cols']
        sel_l = sel.tolist()
        cols = [cols_all[j] for j in sel_l]
        sc = rnd.screen
        c3 = rnd._cam3
        if c3 is None:
            for k, pts in project_polys(rnd, kg['flat'][sel]):
                pygame.draw.polygon(sc, cols[k], pts)
            return
        # the two long faces; each is drawn only when it faces the eye and
        # the block is near enough for the face to be a pixel tall
        near = sel[c3.ground_depth(kg['mid'][sel]) < KERB_FACE_DEPTH]
        if len(near) and self.detail != 'low':
            eye = c3.eye[:2]
            fq, fk = [], []
            for fc in kg['faces']:
                vis = near[((eye - fc['mid'][near]) * fc['nrm'][near]).sum(axis=1) > 0.0]
                if len(vis):
                    fq.append(fc['q'][vis])
                    fk += vis.tolist()
            if fq:
                fcol = [kg['fcols'][j] for j in fk]
                for k, pts in project_polys(rnd, np.concatenate(fq)):
                    pygame.draw.polygon(sc, fcol[k], pts)
        for k, pts in project_polys(rnd, kg['tops'][sel]):
            pygame.draw.polygon(sc, cols[k], pts)

    # ------------------------------------------------------------------ #
    #  LONG LINES (render._draw_edges hands over to this)                 #
    # ------------------------------------------------------------------ #
    def lines(self, rnd, runs) -> None:
        """The painted long lines as perspective strips (a 0.14 m line is 26
        px at 5 m and one pixel at 100 m, as paint is): the edge lines inset
        EDGE_INSET from the edge and broken where a kerb takes over the edge,
        and the per-map ones (the skidpad's driven circle, the drag lane)."""
        tr = self.track
        lay = self.lay
        n = self._n
        polys, cols = [], []
        for idx in runs:
            if len(idx) < 2:
                continue
            u = _unwrap(idx, n, tr.closed)
            for ln in lay.lines:
                kb = ln['kerb_break']
                segs = [idx]
                if kb:
                    kmask = (lay.kerbL if kb > 0 else lay.kerbR)
                    full = np.arange(u[0], u[-1] + 1)
                    fm = full % n if tr.closed else full
                    km = kmask[fm]
                    if km.any():
                        # keep the decimated samples plus every kerb boundary,
                        # then split into the stretches with no kerb
                        tt = np.flatnonzero(np.diff(km.astype(np.int8)))
                        extra = np.concatenate([full[tt], full[np.minimum(tt + 1, len(full) - 1)]])
                        uu = np.union1d(u, extra)
                        mk = kmask[uu % n if tr.closed else uu]
                        d = np.diff(np.concatenate([[1], mk.astype(np.int8), [1]]))
                        st, en = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
                        segs = [(uu[a:b] % n if tr.closed else uu[a:b]) for a, b in zip(st, en)
                                if b - a >= 2]
                for ii in segs:
                    polys.append((ii, ln['n'], ln['w'], PAINT_RGB[ln['key']]))
        self._paint_lines(rnd, polys)

    def _paint_lines(self, rnd, items) -> None:
        """[(sample indices, n offset (scalar or per sample), width m,
        colour)] -> perspective strips
        where the paint is at least LINE_POLY_PX thick ON SCREEN, polylines
        (2 px, then 1 px) where it is thinner.

        Thickness is measured, per sample, as the screen distance between the
        strip's two boundary points: a line running ACROSS the view is
        foreshortened by the ground's slope (w * fl * h / d^2 -- 0.2 px for
        0.15 m at 40 m), so a lateral px/m test would leave it a sliver that
        pygame fills as dashes. Plan view: a polyline at the paint's px width,
        or a strip where that is wider than LINE_POLY_PX."""
        if not items:
            return
        tr = self.track
        sc = rnd.screen
        c3 = rnd._cam3
        polys, pcols, plines = [], [], []
        if c3 is None:
            for ii, n_, w, col in items:
                P, nr = tr.xy[ii], self._nrm[ii]
                n_ = np.asarray(n_, dtype=np.float64)
                n_ = n_[:, None] if n_.ndim else n_
                wpx = w * rnd.ppm
                if wpx >= LINE_POLY_PX:
                    h = 0.5 * w
                    polys.append(np.vstack([P + (n_ + h) * nr, (P + (n_ - h) * nr)[::-1]]))
                    pcols.append(col)
                else:
                    S = rnd.world_to_screen(P + n_ * nr)
                    plines.append((S, col, max(1, int(round(wpx)))))
        else:
            Ls, Rs = [], []
            for ii, n_, w, col in items:
                P, nr = tr.xy[ii], self._nrm[ii]
                n_ = np.asarray(n_, dtype=np.float64)
                n_ = n_[:, None] if n_.ndim else n_
                h = 0.5 * w
                Ls.append(P + (n_ + h) * nr)
                Rs.append(P + (n_ - h) * nr)
            Lb, Rb = np.vstack(Ls), np.vstack(Rs)
            m = len(Lb)
            Q = c3.camera(_as3(np.vstack([Lb, Rb])))
            pz = Q[:, 2]
            pzc = np.maximum(pz, Z_DIRECT)
            S = np.column_stack([0.5 * rnd.W + c3.fl * Q[:, 0] / pzc,
                                 0.5 * rnd.H - c3.fl * Q[:, 1] / pzc])
            SL, SR = S[:m], S[m:]
            thick = np.hypot(SL[:, 0] - SR[:, 0], SL[:, 1] - SR[:, 1])
            near = np.minimum(pz[:m], pz[m:]) < 4.0 * Z_DIRECT
            zone_all = np.where(near | (thick >= LINE_POLY_PX), 0, np.where(thick >= 1.0, 1, 2))
            # the strips (zone 0) are cut by depth as well (_depth_pieces:
            # a far vertex costs every row the strip spans)
            band = np.digitize(np.minimum(pz[:m], pz[m:]), STRIP_SPLIT_D)
            key_all = np.where(zone_all == 0, band, 10 + zone_all)
            SC = 0.5 * (SL + SR)
            o = 0
            for ii, n_, w, col in items:
                k = len(ii)
                zone = zone_all[o:o + k]
                cut = np.flatnonzero(np.diff(key_all[o:o + k])) + 1
                for a, b in zip(np.concatenate([[0], cut]), np.concatenate([cut, [k]])):
                    b2 = min(b + 1, k)
                    if b2 - a < 2:
                        continue
                    z = int(zone[a])
                    if z == 0:
                        polys.append(np.vstack([Lb[o + a:o + b2], Rb[o + a:o + b2][::-1]]))
                        pcols.append(col)
                    else:
                        plines.append((SC[o + a:o + b2], col, 2 if z == 1 else 1))
                o += k
        for k, pts in project_polys(rnd, polys):
            pygame.draw.polygon(sc, pcols[k], pts)
        W, H = rnd.W, rnd.H
        for S_, col, w in plines:
            if len(S_) < 2:
                continue
            if (S_[:, 0].max() < 0 or S_[:, 0].min() > W or S_[:, 1].max() < 0
                    or S_[:, 1].min() > H):
                continue
            if w <= 1:
                # the farthest paint: anti-aliased, or a 1 px line near the
                # horizontal stair-steps into what reads as a dashed line
                pygame.draw.aalines(sc, col, False, S_.tolist())
            else:
                pygame.draw.lines(sc, col, False, S_.astype(np.int32).tolist(), w)

    # ------------------------------------------------------------------ #
    def dashes(self, rnd, windows) -> None:
        """Centre dashes (3 m on / 6 m off) where the map keeps them (the
        open map's perimeter road), as perspective quads, ONE projection."""
        if not self.lay.dashes:
            return
        tr = self.track
        ss = []
        for s0, s1 in windows:
            ss.append(np.arange(math.floor(s0 / 9.0) * 9.0, s1, 9.0))
        if not ss:
            return
        s = np.concatenate(ss)
        if not len(s):
            return
        Q = scn.quads_at(tr, s, s + 3.0, -0.06, 0.06, self._nrm)
        self._paint_quads(rnd, Q, [PAINT_RGB[1]] * len(Q))

    # ------------------------------------------------------------------ #
    def marks(self, rnd, windows) -> None:
        """The start chequer, the sector lines (painted white, not the HUD's
        purple), the per-map paint (grid boxes, the drag launch box, gate
        bars and distance numerals) -- one projection for all of it."""
        tr = self.track
        lay = self.lay
        R_ = _R()
        hw = 0.5 * tr.width
        L = tr.length
        quads, cols = [], []
        ns = np.linspace(-hw, hw, 9)
        for s0, s1 in windows:
            for k, s_line in enumerate(tr.sector_s):
                # exact-s corners (scenery.quads_at): the sector bar is
                # SECTOR_BAR long, as render's own purple bar is, not the
                # 0 / 0.5 m the nearest sample made of a 0.25 m bar (a
                # 1 px hairline at arena S2 and both open-map sectors)
                ln = 1.2 if k == 0 else SECTOR_BAR
                hit = rnd._overlaps(s_line, s_line + ln, s0, s1)
                if hit is None:
                    continue
                a, b = hit
                if k == 0:
                    quads.append(scn.quads_at(tr, a, b, ns[:-1], ns[1:], self._nrm))
                    cols += [R_.C_EDGE if j % 2 == 0 else (40, 42, 46) for j in range(8)]
                else:
                    quads.append(scn.quads_at(tr, a, b, -hw + 0.3, hw - 0.3, self._nrm))
                    cols.append(SECTOR_RGB)
        if len(lay.paint):
            m = _in_windows(lay.paint_s, windows, L, tr.closed, pad=4.0)
            if m.any():
                quads.append(lay.paint[m])
                cols += [PAINT_RGB[int(c)] for c in lay.paint_col[m]]
        #  task 45, round 3: a running stop challenge's brake marker and board
        #  (render.draw_frame keeps HudData.stop_board as `_stop_board`)
        sb = getattr(rnd, "_stop_board", None)
        if sb is not None:
            s0_, s1_, n0_, n1_, k_ = scn.stop_board_rects(tr, sb)
            m = _in_windows(0.5 * (s0_ + s1_), windows, L, tr.closed, pad=2.0)
            if m.any():
                quads.append(scn.quads_at(tr, s0_[m], s1_[m], n0_[m], n1_[m], self._nrm))
                cols += [PAINT_RGB[int(c)] for c in k_[m]]
        if quads:
            self._paint_quads(rnd, np.concatenate(quads), cols)
        if sb is not None and rnd._cam3 is not None:
            self._stop_uprights(rnd, sb)

    def _stop_uprights(self, rnd, sb) -> None:
        """Chase only (task 45, round 3): the stop board's uprights, so the
        brake marker and the board read from the start, where their paint
        is a pixel deep -- an amber cone either side of the race lane at the
        marker, and a black-and-white checker board on a post either side at
        the board's line, 0.9 m outside the race lane: well inside the strip,
        so the walls (props, drawn after this layer) stand too far out to
        cover them. Each face that looks at the eye, far to near, hazed by
        depth, one projection."""
        tr = self.track
        eye = np.asarray(rnd._cam3.eye, dtype=np.float64)
        xy, nv = scn.frame_at(tr, np.array([float(sb[0]), float(sb[1])]), self._nrm)
        faces = []                             # (depth, (M,3) polygon, rgb)
        amber, dark = PAINT_RGB[scn.PAINT_MARKER], PAINT_RGB[scn.PAINT_BOARD_DARK]
        for sgn in (1.0, -1.0):
            n_c = sgn * (scn.DRAG_LANE_HALF + 0.9)
            # the cone: a square pyramid, 0.46 m base, 0.72 m tall
            P, N = xy[0] + n_c * nv[0], nv[0]
            T = np.array([N[1], -N[0]])
            apex = np.array([P[0], P[1], 0.72])
            b = [P + 0.23 * (T * u + N * v) for u, v in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            for j in range(4):
                a0, a1 = b[j], b[(j + 1) % 4]
                tri = np.array([[a0[0], a0[1], 0.0], [a1[0], a1[1], 0.0], apex])
                c = tri.mean(axis=0)
                out = np.array([*(0.5 * (a0 + a1) - P), 0.0])
                if float(out @ (eye - c)) <= 0.0:
                    continue                   # a face turned away
                lit = abs(float(out[:2] @ T)) > abs(float(out[:2] @ N))
                faces.append((float(np.linalg.norm(c - eye)), tri,
                              amber if lit else tuple(int(0.7 * v) for v in amber)))
            # the board: a 1.2 x 1.2 m checker on a post, square to the strip
            P, N = xy[1] + n_c * nv[1], nv[1]
            for i, (z0, z1) in enumerate(((0.45, 1.05), (1.05, 1.65))):
                for j, (w0, w1) in enumerate(((-0.6, 0.0), (0.0, 0.6))):
                    A, B = P + w0 * N, P + w1 * N
                    q = np.array([[A[0], A[1], z0], [B[0], B[1], z0],
                                  [B[0], B[1], z1], [A[0], A[1], z1]])
                    faces.append((float(np.linalg.norm(q.mean(axis=0) - eye)) + 0.01, q,
                                  PAINT_RGB[0] if (i + j) % 2 else dark))
            A, B = P - 0.05 * N, P + 0.05 * N
            q = np.array([[A[0], A[1], 0.0], [B[0], B[1], 0.0], [B[0], B[1], 0.45],
                          [A[0], A[1], 0.45]])
            faces.append((float(np.linalg.norm(q.mean(axis=0) - eye)), q, dark))
        faces.sort(key=lambda f: -f[0])        # far to near
        drawn = project_polys(rnd, [f[1] for f in faces])
        sc = rnd.screen
        from .props import FORBIDDEN           # a colour a self-check counts: nudged, as props do
        counted = {tuple(int(v) for v in c) for c in FORBIDDEN}
        for k, pts in drawn:
            if len(pts) >= 3:
                c = hazed(faces[k][2], faces[k][0])
                if c in counted:
                    c = (c[0], c[1], c[2] + 1 if c[2] < 255 else 254)
                pygame.draw.polygon(sc, c, pts)

    def _paint_quads(self, rnd, Q, cols) -> None:
        """Road-frame paint rectangles (k,4,2), ONE projection. A quad
        thinner on screen than THIN_PX in either direction is drawn as a
        1 px line down its long axis, its colour mixed toward the tarmac by
        the fraction of a pixel it covers: pygame's fill of a sub-pixel
        sliver drops whole runs of it, and a 0.4 m bar seen from 2 m up at
        40 m is 0.5 px tall -- it read as a dotted hairline."""
        sc = rnd.screen
        drawn = project_polys(rnd, np.asarray(Q, dtype=np.float64))
        if not drawn:
            return
        # thickness of every 4-vertex result at once: the distances between
        # the midpoints of opposite edges (01 vs 23, 12 vs 30); the thin
        # ones' lines and coverage colours, vectorised too
        four = [j for j, (_k, pts) in enumerate(drawn) if len(pts) == 4]
        thin = {}
        if four:
            A = np.array([drawn[j][1] for j in four], dtype=np.float64)     # (m,4,2)
            ta = 0.5 * np.hypot(*(A[:, 0] + A[:, 1] - A[:, 2] - A[:, 3]).T)
            tb = 0.5 * np.hypot(*(A[:, 1] + A[:, 2] - A[:, 3] - A[:, 0]).T)
            t = np.minimum(ta, tb)
            th = np.flatnonzero(t < THIN_PX)
            if len(th):
                At, a_ = A[th], (ta[th] <= tb[th])[:, None]
                e0 = np.where(a_, 0.5 * (At[:, 3] + At[:, 0]), 0.5 * (At[:, 0] + At[:, 1]))
                e1 = np.where(a_, 0.5 * (At[:, 1] + At[:, 2]), 0.5 * (At[:, 2] + At[:, 3]))
                f = np.clip(t[th], 0.35, 1.0)[:, None]
                tm = np.array(_R().C_TARMAC, np.float64)
                cc = np.array([cols[drawn[four[m_]][0]] for m_ in th.tolist()], np.float64)
                cf = (tm + (cc - tm) * f).astype(np.int64).tolist()
                for m_, a0, a1, c_ in zip(th.tolist(), e0.tolist(), e1.tolist(), cf):
                    thin[four[m_]] = (a0, a1, c_)
        for j, (k, pts) in enumerate(drawn):
            tj = thin.get(j)
            if tj is None:
                pygame.draw.polygon(sc, cols[k], pts)
            else:
                pygame.draw.line(sc, tj[2], tj[0], tj[1], 1)

    # ------------------------------------------------------------------ #
    #  THE OPEN MAP'S PAD, AND THE PAINTED FEATURES                       #
    # ------------------------------------------------------------------ #
    def areas(self, rnd) -> None:
        """Open map: a verge ring round the pad, the pad, its concrete slab
        joints (8 m, only near the eye -- a joint grid to the horizon is
        moire, not detail), and the wet square with its sky streaks."""
        R_ = _R()
        sc = rnd.screen
        r = rnd._view_radius()
        cx, cy = rnd.cam[0], rnd.cam[1]
        polys, cols, outline = [], [], []
        if self._pad is not None:
            polys.append(self._pad['annulus'])
            cols.append(VERGE_RGB)
            outline.append(False)
        for area, poly, bb in rnd._areas:
            if bb[2] < cx - r or bb[0] > cx + r or bb[3] < cy - r or bb[1] > cy + r:
                continue
            wet = area.mu_scale < 0.95
            if wet:
                x0, y0, x1, y1 = bb
                polys.append(np.array([(x0 - 0.5, y0 - 0.5), (x1 + 0.5, y0 - 0.5),
                                       (x1 + 0.5, y1 + 0.5), (x0 - 0.5, y1 + 0.5)]))
                cols.append(WATER_FRINGE)
                outline.append(False)
            polys.append(poly)
            pad = area.drivable and area.kind == 'rrect'
            cols.append(PAD_RGB if pad else tuple(area.colour))
            outline.append(pad)
        drawn = project_polys(rnd, polys)
        for k, pts in drawn:
            pygame.draw.polygon(sc, cols[k], pts)
            if outline[k]:
                pygame.draw.polygon(sc, R_.C_EDGE, pts, max(1, int(round(0.12 * rnd.ppm))))
            if outline[k] and self.detail != 'low':
                self._joints(rnd)
        aq = self._water['area_quads']
        if len(aq) and self.detail != 'low':
            m = self._cull_pts(rnd, aq.mean(axis=1), 4.0)
            self._streaks(rnd, aq[m])

    def _joints(self, rnd) -> None:
        pad = self._pad
        if pad is None:
            return
        a = pad['area']
        cxa, cya, hx, hy, rr = a.params
        c3 = rnd._cam3
        rad = 110.0 if c3 is not None else rnd._view_radius()
        # the disc of joints is centred where the eye looks
        ox, oy = rnd.cam[0], rnd.cam[1]
        if c3 is not None:
            ox, oy = float(c3.eye[0]), float(c3.eye[1])
            f = c3._f
            nf = math.hypot(f[0], f[1]) or 1.0
            ox += f[0] / nf * 0.5 * rad
            oy += f[1] / nf * 0.5 * rad
        pitch = 8.0
        A, B = [], []
        for axis in (0, 1):
            o_along, o_across = (ox, oy) if axis == 1 else (oy, ox)
            c_across = cxa if axis == 0 else cya
            h_across = hx if axis == 0 else hy
            c_along = cya if axis == 0 else cxa
            h_along = hy if axis == 0 else hx
            vs = np.arange(math.ceil((o_across - rad) / pitch) * pitch, o_across + rad, pitch)
            vs = vs[np.abs(vs - c_across) < h_across - 0.5]
            if not len(vs):
                continue
            # the rrect's extent along the joint at this offset
            q = np.abs(vs - c_across) - (h_across - rr)
            ext = (h_along - rr) + np.sqrt(np.clip(rr * rr - np.clip(q, 0.0, None) ** 2, 0.0, None))
            half = np.sqrt(np.clip(rad * rad - (vs - o_across) ** 2, 0.0, None))
            lo = np.maximum(c_along - ext, o_along - half)
            hi = np.minimum(c_along + ext, o_along + half)
            ok = hi > lo
            for v, l_, h_ in zip(vs[ok], lo[ok], hi[ok]):
                if axis == 0:
                    A.append((v, l_))
                    B.append((v, h_))
                else:
                    A.append((l_, v))
                    B.append((h_, v))
        if not A:
            return
        SA, SB, live = project_segs(rnd, np.array(A), np.array(B))
        sc = rnd.screen
        for k in np.flatnonzero(live):
            pygame.draw.line(sc, PAD_JOINT, SA[k], SB[k], 1)

    def features(self, rnd) -> None:
        """The open map's painted features (circles, lines, boxes, cones) and
        the skidpad's guide circles -- every ring and line in ONE segment
        projection, widths by depth in the chase view."""
        sc = rnd.screen
        r = rnd._view_radius() * 1.1
        cx, cy = rnd.cam[0], rnd.cam[1]
        A, B, W_, C_ = [], [], [], []
        boxes, bcols, cones = [], [], []
        rings = [(f[1], f[2], f[3], f[4], f[5]) for f in rnd._features if f[0] == 'circle']
        rings += [(x, y, R, CHALK, 0.15) for x, y, R, _on in self.lay.guide_rings]
        for fx, fy, rad, col, w in rings:
            if (fx - cx) ** 2 + (fy - cy) ** 2 > (r + rad) ** 2:
                continue
            nseg = int(np.clip(rad * 1.2, 24, 160))
            t = np.linspace(0.0, 2 * math.pi, nseg + 1)
            ring = np.column_stack([fx + rad * np.cos(t), fy + rad * np.sin(t)])
            # only the arcs near the view
            mm = (ring[:, 0] - cx) ** 2 + (ring[:, 1] - cy) ** 2 < (r * 1.2) ** 2
            keep = mm[:-1] | mm[1:]
            A.append(ring[:-1][keep])
            B.append(ring[1:][keep])
            W_ += [w] * int(keep.sum())
            C_ += [col] * int(keep.sum())
        for f in rnd._features:
            kind = f[0]
            if kind == 'line':
                _k, x0, y0, x1, y1, col, w = f
                mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                half = 0.5 * math.hypot(x1 - x0, y1 - y0)
                if (mx - cx) ** 2 + (my - cy) ** 2 > (r + half) ** 2:
                    continue
                A.append(np.array([[x0, y0]]))
                B.append(np.array([[x1, y1]]))
                W_.append(w)
                C_.append(col)
            elif kind == 'box':
                _k, x0, y0, x1, y1, col = f
                if (0.5 * (x0 + x1) - cx) ** 2 + (0.5 * (y0 + y1) - cy) ** 2 > r * r:
                    continue
                boxes.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                bcols.append(col)
            elif kind == 'cone':
                _k, fx, fy = f
                if (fx - cx) ** 2 + (fy - cy) ** 2 <= r * r:
                    cones.append((fx, fy))
        if A:
            AA, BB = np.vstack(A), np.vstack(B)
            SA, SB, live = project_segs(rnd, AA, BB)
            ppm = _depth_px(rnd, 0.5 * (AA + BB))
            for k in np.flatnonzero(live):
                wpx = int(min(max(round(W_[k] * ppm[k]), 1), 8))
                pygame.draw.line(sc, C_[k], SA[k], SB[k], wpx)
        if boxes:
            for k, pts in project_polys(rnd, np.array(boxes, dtype=np.float64)):
                pygame.draw.polygon(sc, bcols[k], pts)
        if cones:
            R_ = _R()
            P = np.array(cones, dtype=np.float64)
            c3 = rnd._cam3
            if c3 is not None:
                d = c3.ground_depth(P)
                ok = d > R_.CHASE_GROUND_NEAR
                P, d = P[ok], d[ok]
                S = c3.ground(P) if len(P) else P
                rads = np.maximum(2, np.round(0.22 * c3.fl / np.maximum(d, 0.5))).astype(int)
            else:
                S = rnd.world_to_screen(P)
                rads = np.full(len(P), max(2, int(round(0.22 * rnd.ppm))))
            for (sx, sy), rd in zip(S.tolist() if len(P) else [], rads.tolist()):
                pygame.draw.circle(sc, R_.C_WING_ON, (int(sx), int(sy)), rd)
                if rd >= 4:
                    pygame.draw.circle(sc, (40, 30, 20), (int(sx), int(sy)), rd, 1)

    # ------------------------------------------------------------------ #
    #  SKID MARKS (render._draw_skid hands over to this)                  #
    # ------------------------------------------------------------------ #
    def _surface_grid(self):
        """A 0.5 m raster of what the ground is -- 0 grass, 1 tarmac, 2
        gravel, 3 verge, 4 run-off -- drawn ONCE (2 ms) with the same
        polygons the frame draws, so a skid mark can be coloured by what it
        is on for ~20 us a frame (four `track.project` per mark would be
        4 ms), and the grass tufts kept off everything that is not grass."""
        if self._grid is not None:
            return self._grid
        tr = self.track
        cell = 0.5
        pad = scn.DRESS_MAX + 12.0
        x0, y0, x1, y1 = tr.bbox
        for a in getattr(tr, 'areas', []):
            bx0, by0, bx1, by1 = a.bbox()
            x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
        surf = pygame.Surface((nx, ny), depth=8)
        surf.fill(0)

        def px(P):
            P = np.asarray(P, dtype=np.float64)
            return np.column_stack([(P[:, 0] - x0) / cell, (P[:, 1] - y0) / cell]).tolist()
        if self._pad is not None:
            pygame.draw.polygon(surf, 3, px(self._pad['ring']))
        for a in getattr(tr, 'areas', []):
            if a.drivable:
                pygame.draw.polygon(surf, 1, px(a.polygon(16)))
        N = len(tr.s)
        g = 0.5 * tr.width + scn.VERGE_W
        for a in range(0, N - 1, 40):
            b = min(a + 41, N)
            if self.lay.verge:
                P, nr = tr.xy[a:b], self._nrm[a:b]
                pygame.draw.polygon(surf, 3, px(np.vstack([P + g * nr, (P - g * nr)[::-1]])))
        for a in range(0, N - 1, 40):
            b = min(a + 41, N)
            pygame.draw.polygon(surf, 1, px(np.vstack([tr.left[a:b], tr.right[a:b][::-1]])))
        for poly, kind in zip(self._chunk_poly, self._chunk_kind):
            pygame.draw.polygon(surf, 2 if kind == 'gravel' else 4, px(poly))
        self._grid = (pygame.surfarray.array2d(surf).astype(np.uint8), x0, y0, cell)
        return self._grid

    def _skid_arrays(self, skid) -> list:
        """The four wheel deques as (n, 4) arrays [x, y, t, new_run], kept up
        to date INCREMENTALLY: only the points emitted since the last frame
        are converted (a full 4 x 4000-point buffer is ~4 ms to walk in
        Python every frame, which is what SkidBuffer.visible does)."""
        cache = getattr(self, '_skc', None)
        if cache is None or len(cache) != len(skid.q):
            cache = self._skc = [np.zeros((0, 4)) for _ in skid.q]
        for w, d in enumerate(skid.q):
            c = cache[w]
            n = len(d)
            if n == 0:
                cache[w] = np.zeros((0, 4))
                continue
            if len(c) == n and c[-1, 2] == d[-1][2] and c[0, 2] == d[0][2]:
                continue
            k = 0
            if len(c):
                t_c = c[-1, 2]
                for i in range(n - 1, -1, -1):
                    if d[i][2] <= t_c:
                        break
                    k += 1
            if len(c) and k < n:
                new = np.array([d[i] for i in range(n - k, n)], dtype=np.float64).reshape(-1, 4)
                c = np.vstack([c, new])[-n:]
            if len(c) != n or c[0, 2] != d[0][2] or c[-1, 2] != d[-1][2]:
                c = np.array(d, dtype=np.float64).reshape(-1, 4)    # reset / clear
            cache[w] = c
        return cache

    def _skid_segs(self, skid, cam, radius, max_segs):
        """(P0 (n,2), P1 (n,2), alpha (n,)) -- SkidBuffer.visible's contract
        (eq.16 alpha, fade, cull radius, <= max_segs) but strided by POINTS
        within each run, so a thinned mark is a coarser continuous line
        rather than every other segment missing (it read as dotted)."""
        t_now = float(skid._t_last)
        fade = float(skid.fade_s)
        r2 = float(radius) ** 2
        per = []
        total = 0
        for A in self._skid_arrays(skid):
            if len(A) < 2:
                continue
            x, y, t, nr = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
            fresh = (t_now - t) <= fade
            near = (x - cam[0]) ** 2 + (y - cam[1]) ** 2 <= r2
            live = fresh & near
            rid = np.cumsum((nr > 0.5) | ~live)
            per.append((A, live, rid))
            total += int(live.sum())
        if not per or total < 2:
            return None
        st = max(1, -(-total // max_segs))
        P0s, P1s, als = [], [], []
        inv = 150.0 / fade
        for A, live, rid in per:
            idx = np.flatnonzero(live)
            if len(idx) < 2:
                continue
            rf = rid[idx]
            first = np.r_[True, rf[1:] != rf[:-1]]
            last = np.r_[rf[1:] != rf[:-1], True]
            pos = np.arange(len(idx))
            local = pos - np.maximum.accumulate(np.where(first, pos, 0))
            keep = (local % st == 0) | last
            ki = idx[keep]
            same = rid[ki[1:]] == rid[ki[:-1]]
            a_, b_ = ki[:-1][same], ki[1:][same]
            al = 150.0 - (t_now - A[b_, 2]) * inv
            ok = al > 0.0
            P0s.append(A[a_[ok], :2])
            P1s.append(A[b_[ok], :2])
            als.append(al[ok])
        if not P0s:
            return None
        return np.vstack(P0s), np.vstack(P1s), np.concatenate(als)

    def skid(self, rnd, skid) -> None:
        """<= SKID_MAX_SEGS marks, coloured by what they are on (dark rubber
        on tarmac, churned earth on grass, furrows in gravel), continuous
        when thinned, and in the chase view as wide as a tyre at its depth."""
        R_ = _R()
        got = self._skid_segs(skid, rnd.cam, rnd._view_radius() * 1.15, R_.SKID_MAX_SEGS)
        if got is None:
            return
        P0, P1, al = got
        c3 = rnd._cam3
        if c3 is not None:
            d = np.minimum(c3.ground_depth(P0), c3.ground_depth(P1))
            keep = d > R_.CHASE_GROUND_NEAR
            if not keep.any():
                return
            P0, P1, al, d = P0[keep], P1[keep], al[keep], d[keep]
            wpx = np.clip(np.round(R_.SKID_WIDTH * c3.fl / d), 1, 14).astype(int)
        else:
            wpx = np.full(len(P0), max(1, int(round(R_.SKID_WIDTH * rnd.ppm))), dtype=int)
        g, gx0, gy0, cell = self._surface_grid()
        M = 0.5 * (P0 + P1)
        ix = np.clip(((M[:, 0] - gx0) / cell).astype(int), 0, g.shape[0] - 1)
        iy = np.clip(((M[:, 1] - gy0) / cell).astype(int), 0, g.shape[1] - 1)
        lab = g[ix, iy]
        base = np.array([GRASS_RGB, R_.C_TARMAC, GRAVEL_RGB, VERGE_RGB, RUNOFF_RGB],
                        dtype=np.float64)[lab]
        mark = np.array([MUD_RGB, R_.C_SKID, GRAVEL_SKID, MUD_RGB, R_.C_SKID],
                        dtype=np.float64)[lab]
        # tarmac keeps render's blend (alpha / 255, eq.16); earth is churned
        # harder than rubber is laid, so off the road the mark is stronger
        paved = (lab == 1) | (lab == 4)
        t = np.where(paved, al / 255.0, np.minimum(al / 150.0, 1.0) * 0.85)
        col = (base + (mark - base) * t[:, None]).astype(int).tolist()
        SA, SB, live = project_segs(rnd, P0, P1)
        sc = rnd.screen
        wl = wpx.tolist()
        SAl, SBl = SA.tolist(), SB.tolist()
        for k in np.flatnonzero(live).tolist():
            pygame.draw.line(sc, col[k], SAl[k], SBl[k], wl[k])

    # ------------------------------------------------------------------ #
    #  ATMOSPHERE                                                        #
    # ------------------------------------------------------------------ #
    def draw_atmosphere(self, rnd) -> None:
        """Haze over the ground layers, before the props and the car: ONE
        SRCALPHA band under the horizon whose alpha per row is
        min(haze_factor(the camera depth of the ground on that row),
        HAZE_LAND). Rebuilt only when the horizon row, the eye height or the
        pitch moves (a jolted frame: ~0.05 ms)."""
        c3 = rnd._cam3
        if c3 is None:
            return
        H, W = rnd.H, rnd.W
        yh = float(c3.horizon_y)
        if yh >= H:
            return
        f, u = c3._f, c3._u
        ez = float(c3.eye[2])
        key = (int(round(yh * 2)), int(round(ez * 40)), int(round(f[2] * 4000)), W)
        if key != self._haze_key:
            self._haze_key = key
            y0 = int(math.floor(yh)) - 3
            ys = np.arange(y0, min(H, y0 + 400)) + 0.5
            dz = -(f[2] - ((ys - 0.5 * H) / c3.fl) * u[2])     # -dir_z
            dep = np.where(dz > 1e-9, ez / np.maximum(dz, 1e-9), 1e9)
            # capped at HAZE_LAND: the ground beyond LAND_D is what the
            # panorama's land would hide, so it takes the land's own haze
            # (its foot is LAND_FOOT_RGB) -- one tone where they meet, not
            # a pale strip. Above the horizon: the panorama, untouched.
            a = np.minimum(_haze_factor_v(dep), HAZE_LAND)
            a = np.where(ys < yh, 0.0, a)
            last = np.flatnonzero(a > 1.0 / 255.0)
            n = int(last[-1]) + 1 if len(last) else 0
            if n <= 0:
                self._haze_surf = None
                return
            col = pygame.Surface((1, n), pygame.SRCALPHA)
            col.fill((*HAZE_RGB, 0))
            pa = pygame.surfarray.pixels_alpha(col)
            pa[0, :] = np.clip(np.round(a[:n] * 255.0), 0, 255).astype(np.uint8)
            del pa
            self._haze_surf = pygame.transform.scale(col, (W, n))
            self._haze_y0 = y0
        if self._haze_surf is not None:
            rnd.screen.blit(self._haze_surf, (0, self._haze_y0))


# ======================================================================= #
#  SELF-CHECK                                                              #
# ======================================================================= #
def self_check(verbose: bool = True) -> bool:
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    from . import render as R
    from . import track as trk
    ok_all = True
    res = []

    def rep(tag, passed, msg):
        nonlocal ok_all
        ok_all = ok_all and bool(passed)
        res.append((tag, passed, msg))
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag:46s} {msg}")

    if verbose:
        print('drive/world.py self-check')
    forbidden = {(217, 206, 85), (78, 194, 106), (226, 82, 63), (176, 78, 224), (120, 220, 160)}
    pal = [v for k, v in globals().items() if k.isupper() and isinstance(v, tuple)
           and len(v) == 3 and all(isinstance(x, int) for x in v)]
    pal += [c for k in ('PATCH_RGB', 'PAINT_RGB', 'GROOVE_RGB', 'WATER_SHEEN', 'TUFT_RGB',
                        'TUFT_PLAN_RGB') for c in globals()[k]]
    bad = [c for c in pal if tuple(c) in forbidden]
    rep('palette avoids every colour a self-check counts', not bad,
        f'{len(pal)} colours' + (f', CLASH {bad}' if bad else ''))
    hz = [haze_factor(d) for d in (0.0, HAZE_NEAR, 200.0, HAZE_FAR, 5000.0)]
    hv = _haze_factor_v(np.array([0.0, HAZE_NEAR, 200.0, HAZE_FAR, 5000.0]))
    rep('haze: 0 near, monotone, HAZE_MAX far; vector == scalar',
        hz[0] == 0.0 and hz[1] == 0.0 and 0 < hz[2] < hz[3] == HAZE_MAX == hz[4]
        and np.allclose(hz, hv, atol=1e-12), f'{[round(v, 3) for v in hz]}')
    # clip_bands: the bands tile the quad (odd + even areas == its area)
    C = np.array([[3.0, -2.0], [40.0, 5.0], [31.0, 44.0], [-6.0, 30.0]])
    m = np.array([math.cos(0.4), math.sin(0.4)])

    def area(P):
        x, y = P[:, 0], P[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    tot = sum(area(P) for P in clip_bands(C, m, 7.0, 1)) + sum(
        area(P) for P in clip_bands(C, m, 7.0, 0))
    rep('stripe clipper tiles the view exactly', abs(tot - area(C)) < 1e-6 * area(C),
        f'bands {tot:.4f} m^2 vs quad {area(C):.4f} m^2')
    # the fast projection == the renderer's own clipped projection
    tr = trk.make_arena()
    rnd = R.Renderer(R.ViewConfig(mode='chase'), tr, headless=True)
    i = R.trk_index(tr, 150.0)
    st = R._demo_state(float(tr.xy[i][0]), float(tr.xy[i][1]), float(tr.psi[i]), u=20.0)
    rnd.update_camera(st, 0.0)
    rng = np.random.default_rng(3)
    Q = tr.xy[i][None, None, :] + rng.uniform(-40, 60, size=(300, 4, 2))
    # compare what reaches the SCREEN: the clipper trims at planes 30 % wider
    # than the frame, the fast path projects those vertices directly, so the
    # polygons differ off screen and must not differ on it
    sa = pygame.Surface((rnd.W, rnd.H))
    sb = pygame.Surface((rnd.W, rnd.H))
    sa.fill((0, 0, 0))
    sb.fill((0, 0, 0))
    for k, pts in project_polys(rnd, Q):
        pygame.draw.polygon(sa, (255, 255, 255), pts)
    for k in range(len(Q)):
        ref = rnd._gpoly(Q[k])
        if len(ref) >= 3:
            pygame.draw.polygon(sb, (255, 255, 255), ref)
    ma = pygame.surfarray.array2d(sa) != 0
    mb = pygame.surfarray.array2d(sb) != 0
    n_diff, n_on = int((ma != mb).sum()), int(mb.sum())
    rep('batched projection == Chase3D clipper, on screen', n_diff <= 2e-3 * n_on,
        f'300 random ground quads round the car: {n_on} px drawn, {n_diff} differ')
    km = keep_masks(tr)
    rep('strip decimation', km['far'].sum() < km['mid'].sum() < km['near'].sum() < len(tr.s),
        f"arena samples {len(tr.s) - 1} -> near {km['near'].sum()}, mid {km['mid'].sum()}, "
        f"far {km['far'].sum()}")
    # determinism: two builds of the panorama are identical
    _PANO.clear()
    p1 = build_panorama('arena', 'circuit', rnd._chase.fl)
    a1 = pygame.surfarray.array3d(p1['surf']).copy()
    _PANO.clear()
    p2 = build_panorama('arena', 'circuit', rnd._chase.fl)
    a2 = pygame.surfarray.array3d(p2['surf'])
    col_top, col_hor = a1[:, 5].mean(axis=0), a1[:, PANO_ROWS - 30].mean(axis=0)
    rep('panorama deterministic, sky darker at the top',
        np.array_equal(a1, a2) and col_top[2] > col_top[0] and col_top.sum() < col_hor.sum(),
        f"{p1['Pw']}x{p1['rows']} px in {p1['ms']:.0f} ms; top {col_top.astype(int).tolist()} "
        f"vs near the horizon {col_hor.astype(int).tolist()}")
    # no cloud is cut off by its box: alpha 0 on every border, every map
    edges = {}
    for nm_ in trk.TRACK_ORDER:            # each map's own sky (seeded by name)
        th_ = 'circuit' if nm_ in trk.CIRCUITS else nm_
        edges[nm_] = round(build_panorama(nm_, th_, rnd._chase.fl)['cloud_edge'], 4)
    rep('clouds: no box edge in the sky (border alpha ~0)', max(edges.values()) < 0.01,
        f'largest cloud alpha on a box border: {edges} (was 0.09-0.92 on 6 of 9 arena clouds)')
    # the land's foot is the ground's tone at the horizon (no fog strip)
    foot = a1[:, PANO_ROWS - 1].mean(axis=0)
    d_foot = float(np.abs(foot - np.array(LAND_FOOT_RGB)).sum())
    rep("land's foot fades to the ground's tone at the horizon", d_foot < 40.0,
        f'bottom row mean {foot.astype(int).tolist()} vs LAND_FOOT_RGB {list(LAND_FOOT_RGB)} '
        f'(ground at haze {HAZE_LAND:.2f})')
    # a lens change never rebuilds the panorama inside a frame, and the
    # detail level is read every frame
    rl = R.Renderer(R.ViewConfig(mode='chase'), tr, headless=True)
    built = []
    real_build = globals()['build_panorama']

    def spy(*a_, **k_):
        built.append(a_)
        return real_build(*a_, **k_)
    globals()['build_panorama'] = spy
    t_bd = []
    try:
        for k in range(40):
            j = R.trk_index(tr, 120.0 + 0.6 * k)
            s_ = R._demo_state(float(tr.xy[j][0]), float(tr.xy[j][1]), float(tr.psi[j]),
                               u=1.2 * k)
            rl.update_camera(s_, 1 / 60.0 if k else 0.0)
            c3_ = rl._chase                         # the lens changes every frame
            if hasattr(c3_, 'set_fov'):
                c3_.set_fov(46.0 + 0.1 * k)
            else:
                c3_.fl = 0.5 * c3_.H / math.tan(math.radians(0.5 * (46.0 + 0.1 * k)))
            t0 = time.perf_counter()
            rl._world.draw_backdrop(rl)
            t_bd.append((time.perf_counter() - t0) * 1e3)
        rl.cfg.detail = 'low'
        rl._world.draw_backdrop(rl)
        det = rl._world.detail
    finally:
        globals()['build_panorama'] = real_build
    rep('a lens change: no rebuild in a frame; detail read live',
        not built and det == 'low' and max(t_bd[1:]) < 5.0,
        f'{len(built)} rebuilds over 40 frames with the FOV 46 -> 50 deg; backdrop '
        f'{np.median(t_bd):.2f} ms median, {max(t_bd[1:]):.2f} max; cfg.detail -> {det}')
    # per-layer cost, world on, chase and plan
    t_c, t_p = [], []
    for mode, acc in (('chase', t_c), ('car_up', t_p)):
        rw = R.Renderer(R.ViewConfig(mode=mode), tr, headless=True)
        for k in range(40):
            j = R.trk_index(tr, 120.0 + 0.45 * k)
            s_ = R._demo_state(float(tr.xy[j][0]), float(tr.xy[j][1]), float(tr.psi[j]), u=27.0)
            rw.update_camera(s_, 1 / 60.0 if k else 0.0)
            wd = rw._world
            t0 = time.perf_counter()
            wd.draw_backdrop(rw)
            wd.draw_ground(rw)
            runs, _s, wins = rw._visible_indices(s_.X, s_.Y)
            wd.ribbon(rw, runs)
            wd.draw_surface(rw, runs, wins)
            wd.kerbs(rw, wins)
            wd.lines(rw, runs)
            wd.marks(rw, wins)
            wd.draw_atmosphere(rw)
            if k >= 5:
                acc.append((time.perf_counter() - t0) * 1e3)
    rep('world layers cost (arena T1, 35 frames)', True,
        f'chase {np.mean(t_c):.2f} ms, plan {np.mean(t_p):.2f} ms (load-dependent; budgets '
        f'are asserted by render.py\'s _world_checks)')
    if verbose:
        print(f"  {'PASS' if ok_all else 'FAIL'}: {sum(1 for r in res if r[1])}/{len(res)} checks")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if self_check() else 1)
