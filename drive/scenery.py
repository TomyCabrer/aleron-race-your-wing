"""drive/scenery.py -- the layout of the ground beside the road.

Pure numpy + `track`: no pygame. WHERE things are -- the kerbs, the verge,
the gravel traps and the painted run-off, the rubbered racing line, the
repair patches and sealed cracks, the grid boxes, the per-map paint -- all
derived GENERICALLY from the track's own geometry (its centreline samples,
curvature and width), so a generated track gets a dressed circuit too. How
it is drawn is `drive/world.py` (and the track layers of `render.py`); what
each wheel is on is `wheel_surfaces`, read by the harness into HudData.surf4
for the effects and the sound. NEVER read by the physics: a kerb here is
paint, the physics' surface is `track.surface_at` and nothing else.

Why a layout module at all: the drawing and `wheel_surfaces` must agree --
a wheel reported 'gravel' has to be in a trap the driver can SEE -- and the
only way two consumers cannot drift is that both read the same tables. So
every region is defined in the track's (s, n) frame, per centreline sample:
the renderer offsets the samples to draw it, `wheel_surfaces` projects a
contact point (`track.project`, 6-7 us measured) and indexes the same arrays.

Per-map themes by `track.name` (anything else is a circuit):
  circuit    inset edge lines, inside + apex + exit kerbs, gravel outside
             the slow corners (R < GRAVEL_R_MAX), painted run-off outside the
             fast ones, rubbered racing line, repair patches, cracks, grid
  open       the proving ground: the pad is tarmac all round the road, so no
             verge and no traps along it; a centre-dashed perimeter road
  skidpad    the guide circles (Track.guide_radii about Track.centre), the
             driven radius rubbered, no exits so no exit kerbs
  dragstrip  a race lane centred on the launch line, the launch box, the
             drag groove (rubber in the wheel tracks), distance paint, and
             a sand trap past the end of the strip

Everything random is seeded from the track NAME (zlib.crc32), so the same
track draws the same patches every run: no wall clock, no unseeded RNG.

THE RUN-OFF IS NOT GRIP. Beyond the ribbon the physics gives every point the
off-track surface (`track.MU_OFF_TRACK` 0.55, Crr x25) whatever is painted
there. So the run-off is painted as ABRASIVE run-off (green bands on a pale
grey), which reads as "not the road", and `wheel_surfaces` reports it as
'grass' -- the rule is 'grass' everywhere off the tarmac that is not a trap.
"""
from __future__ import annotations

import math
import time
import weakref
import zlib

import numpy as np

from . import track as trk

SURFACES = ("tarmac", "wet", "kerb", "grass", "gravel")
THEMES = ("circuit", "open", "skidpad", "dragstrip")

# --- the dressing. All est: the look of a well-kept European circuit, not a
#     survey of one. Distances are from the track EDGE unless stated. -------
VERGE_W = 1.4            # m   est  the mown verge band outside each edge
KERB_W = 0.8             # m   render._draw_kerbs' kerb width, kept verbatim
KERB_BLOCK = 3.0         # m   one red / white block, kept verbatim
KERB_K_FULL = 1.0 / 60.0 # 1/m render's inside-kerb rule (|kappa| > 1/60), kept
KERB_R_APEX = 200.0      # m   est  bends up to this radius carry an apex kerb
APEX_FRAC = 0.30         # -   est  ... over +-30 % of the arc round its middle
EXIT_KERB_BACK = 18.0    # m   est  exit kerb starts this far before the arc
                         #          ends (at most 30 % of the arc) ...
EXIT_KERB_ON = 12.0      # m   est  ... and runs this far onto the straight
CORNER_R_MAX = 400.0     # m   est  a bend gentler than this is not a corner
CORNER_MERGE = 6.0       # m   est  same-hand arcs closer than this are one
GRAVEL_R_MAX = 60.0      # m   est  tighter corners get gravel, faster run-off
DRESS_MAX = 24.0         # m   every trap / run-off stays within this of the
                         #     edge: drive/props.py places solids at >= 30 m
GRAVEL_OVERRUN = 30.0    # m   est  trap runs on past the exit (cars run wide)
GRAVEL_LEAD = 6.0        # m   est  ... and starts a little before the arc
RUNOFF_OVERRUN = 25.0    # m   est
RUNOFF_LEAD = 10.0       # m   est
CHUNK_M = 20.0           # m   traps / run-off are drawn as chunks this long:
                         #     a chunk straddling the eye is clipped alone
EDGE_INSET = 0.30        # m   the white edge line's centre, inside the edge
EDGE_LINE_W = 0.14       # m   est  (circuits paint 0.10-0.20 m)
RL_HALF = 1.05           # m   half-width of the rubbered racing line
RL_MARGIN = 1.6          # m   the line's centre is kept this far in from the
                         #     edge: a 1.65 m car with its wheels on the paint
RL_KNOT = 4.0            # m   knot spacing of the min-curvature solve ...
RL_KNOTS_MAX = 400       # -   ... but never more knots than this
PATCH_EVERY = 150.0      # m   est  one asphalt repair per this much track
CRACK_EVERY = (16.0, 34.0)  # m  est  sealed cracks, spacing range
GRID_ROW_M = 7.0         # m   grid rows this far apart, back from the line:
                         #     where the race lines its cars up (the game's
                         #     race_grid: ROW_M, rows of two at s = 0, -7, ...)
GRID_SIDE_N = 2.2        # m   ... either side of n = 0 (race_grid's
                         #     START_OFFSET_M); your car on n = 0 at s = 0
GRID_ROWS = 6            # -   at most this many rows painted (5 bots use 3)
GRID_R_MIN = 80.0        # m   est  no grid row in a bend tighter than this
GRID_NOSE = 1.75         # m   a grid car's nose ahead of its CG, and its tail
GRID_TAIL = 2.07         # m   behind (render.CAR_X_FRONT / CAR_X_REAR)
GRID_BOX_HALF = 1.0      # m   est  half a box: a 1.65 m car + 0.18 m either
                         #     side; slots 2.2 m apart keep a 0.2 m gap
CHEQUER_L = 1.2          # m   the start chequer's length (render._draw_marks)
STRIPE_W = 9.0           # m   est  mowing-stripe width
DRAG_LANE_HALF = 2.6     # m   est  the dragstrip race lane, either side of n=0
DRAG_LAUNCH_BOX = 14.0   # m   est  the launch box, from the start line
DRAG_GROOVE = ((0.0, 60.0, 0), (60.0, 200.0, 1), (200.0, 420.0, 2))  # s0,s1,shade
TRAP_END = 24.0          # m   the dragstrip's sand trap past the end

# colour KEYS: world.py owns the RGB (the palette), this module owns where
PAINT_WHITE, PAINT_DIM, PAINT_RUBBER = 0, 1, 2


# ======================================================================= #
#  small geometry helpers                                                  #
# ======================================================================= #
def _nper(tr) -> int:
    return (len(tr.s) - 1) if tr.closed else len(tr.s)


def _nrm(tr) -> np.ndarray:
    return np.column_stack([-np.sin(tr.psi), np.cos(tr.psi)])


def _i_of_s(tr, s):
    """Sample index of arclength s (scalar or array), wrapped on a closed
    track, clamped on an open one -- render.Renderer._i_of_s, vectorised."""
    i = np.rint(np.asarray(s, dtype=np.float64) / tr.ds).astype(np.int64)
    if tr.closed:
        return np.mod(i, _nper(tr))
    return np.clip(i, 0, len(tr.s) - 1)


def frame_at(tr, s, nrm=None):
    """(xy (k,2), unit normal (k,2)) at EXACT arclengths s -- interpolated
    between the two samples either side, wrapped on a closed track, clamped
    on an open one. `nrm` is the per-sample normal table when the caller
    keeps one (World does), else it is built.

    Why not `_i_of_s`: that snaps s to the nearest 0.5 m sample, so a paint
    quad shorter than ~0.5 m along the road came out zero-length or 0.5 m
    long -- the 0.20 m grid-box bars and the 0.32 m numeral strokes were
    hairlines (8/24 arena and 23/184 dragstrip quads measured zero-length)."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    ds = float(tr.ds)
    N = len(tr.s)
    if tr.closed:
        u = np.mod(s, _nper(tr) * ds) / ds
    else:
        u = np.clip(s, 0.0, (N - 1) * ds) / ds
    i = np.minimum(np.floor(u).astype(np.int64), N - 2)
    t = (u - i)[:, None]
    xy = tr.xy[i] * (1.0 - t) + tr.xy[i + 1] * t
    nr_ = _nrm(tr) if nrm is None else nrm
    nv = nr_[i] * (1.0 - t) + nr_[i + 1] * t
    nv /= np.maximum(np.hypot(nv[:, 0], nv[:, 1]), 1e-12)[:, None]
    return xy, nv


def _frame1(tr, s: float, nrm=None) -> tuple:
    """frame_at for ONE arclength, in plain floats: (x, y, nx, ny)."""
    ds = float(tr.ds)
    N = len(tr.s)
    if tr.closed:
        u = (s % (_nper(tr) * ds)) / ds
    else:
        u = min(max(s, 0.0), (N - 1) * ds) / ds
    i = min(int(math.floor(u)), N - 2)
    t = u - i
    p0, p1 = tr.xy[i], tr.xy[i + 1]
    if nrm is None:
        q0 = (-math.sin(tr.psi[i]), math.cos(tr.psi[i]))
        q1 = (-math.sin(tr.psi[i + 1]), math.cos(tr.psi[i + 1]))
    else:
        q0, q1 = nrm[i], nrm[i + 1]
    nx, ny = q0[0] * (1.0 - t) + q1[0] * t, q0[1] * (1.0 - t) + q1[1] * t
    h = math.hypot(nx, ny) or 1.0
    return (float(p0[0]) * (1.0 - t) + float(p1[0]) * t, float(p0[1]) * (1.0 - t) + float(p1[1]) * t,
            float(nx) / h, float(ny) / h)


def quads_at(tr, s0, s1, n0, n1, nrm=None) -> np.ndarray:
    """(k,4,2) road-frame rectangles [s0,s1] x [n0,n1] from exact-s points
    (frame_at), vertex order (s0,n0) (s0,n1) (s1,n1) (s1,n0)."""
    if np.ndim(s0) == 0 and np.ndim(s1) == 0:
        # one bar (a sector line, the chequer): plain floats, ~4 us instead
        # of ~25 us of small-array numpy a frame
        (x0, y0, a0, b0), (x1, y1, a1, b1) = _frame1(tr, float(s0), nrm), _frame1(tr, float(s1), nrm)
        n0 = np.atleast_1d(np.asarray(n0, dtype=np.float64))
        n1 = np.atleast_1d(np.asarray(n1, dtype=np.float64))
        n0, n1 = np.broadcast_arrays(n0, n1)
        out = np.empty((len(n0), 4, 2))
        out[:, 0, 0], out[:, 0, 1] = x0 + n0 * a0, y0 + n0 * b0
        out[:, 1, 0], out[:, 1, 1] = x0 + n1 * a0, y0 + n1 * b0
        out[:, 2, 0], out[:, 2, 1] = x1 + n1 * a1, y1 + n1 * b1
        out[:, 3, 0], out[:, 3, 1] = x1 + n0 * a1, y1 + n0 * b1
        return out
    s0, s1, n0, n1 = (np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (s0, s1, n0, n1))
    k = max(len(s0), len(s1), len(n0), len(n1))
    s0, s1, n0, n1 = (np.broadcast_to(v, (k,)) for v in (s0, s1, n0, n1))
    P, N = frame_at(tr, np.concatenate([s0, s1]), nrm)      # one pass, both ends
    P0, P1, N0, N1 = P[:k], P[k:], N[:k], N[k:]
    n0, n1 = n0[:, None], n1[:, None]
    return np.stack([P0 + n0 * N0, P0 + n1 * N0, P1 + n1 * N1, P1 + n0 * N1], axis=1)


def theme_of(tr) -> str:
    """'circuit' | 'open' | 'skidpad' | 'dragstrip' from the track's name."""
    name = str(getattr(tr, "name", ""))
    return name if name in ("open", "skidpad", "dragstrip") else "circuit"


def rng_for(tr, salt: str = "") -> np.random.Generator:
    """The one RNG of a track's dressing: seeded from its NAME, so a layout
    is a pure function of the track (and a generated track, named, too)."""
    return np.random.default_rng(zlib.crc32((str(tr.name) + salt).encode()))


def corners(tr) -> list:
    """Contiguous bends of one hand, from the centreline curvature.

    A dict per corner: i0 (first sample), n (samples), s0, s1 (s1 may exceed
    the lap on a closed track), sgn (+1 left: the inside is +n), R (the
    tightest radius), turn (rad). Arcs of the same hand closer than
    CORNER_MERGE are one corner; slivers shorter than 8 m are dropped.
    """
    n = _nper(tr)
    k = np.asarray(tr.kappa[:n], dtype=np.float64)
    ds = float(tr.ds)
    on = np.abs(k) > 1.0 / CORNER_R_MAX
    sg = np.where(on, np.sign(k), 0).astype(np.int8)
    if not on.any():
        return []
    if tr.closed and on.all():
        # one bend all the way round (the skidpad): a single corner
        return [dict(i0=0, n=n, s0=0.0, s1=n * ds, sgn=int(sg[0]),
                     R=1.0 / float(np.abs(k).max()), turn=float(k.sum() * ds))]
    rot = int(np.flatnonzero(~on)[0]) if tr.closed else 0
    sr = np.roll(sg, -rot)
    runs = []                                  # [start, end) in rolled index
    j = 0
    while j < n:
        if sr[j] == 0:
            j += 1
            continue
        a = j
        while j < n and sr[j] == sr[a]:
            j += 1
        runs.append([a, j, int(sr[a])])
    merged = []
    gap = int(round(CORNER_MERGE / ds))
    for r in runs:
        if merged and r[2] == merged[-1][2] and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for a, b, sgn in merged:
        if (b - a) * ds < 8.0:
            continue
        idx = (np.arange(a, b) + rot) % n
        kk = k[idx]
        i0 = int(idx[0])
        out.append(dict(i0=i0, n=int(b - a), s0=i0 * ds, s1=(i0 + (b - a)) * ds,
                        sgn=int(sgn), R=1.0 / float(np.abs(kk).max()),
                        turn=float(kk.sum() * ds)))
    return out


# ======================================================================= #
#  the rubbered racing line: a MIN-CURVATURE path inside the edges        #
# ======================================================================= #
def _racing_line(tr, hw: float) -> np.ndarray:
    """Lateral offset (N,) of a min-curvature line, |n| <= hw - RL_MARGIN.

    Minimises the summed squared second difference of the path over knots
    RL_KNOT apart -- the classic outside / apex / outside line falls out of
    it on its own, which is what makes the rubber read as a LINE rather
    than a centre stripe. Bounds by an active-set loop on the dense normal
    equations: 312 knots on the arena, ~10 solves, ~40 ms, once per track.
    The dense solve is O(knots^3), so a long track gets sparser knots (at
    most RL_KNOTS_MAX): a 6.2 km generated circuit at 4 m was 1560 knots
    and 1.47 s of layout; a racing line needs no 4 m detail on a long bend.
    """
    n = _nper(tr)
    if not tr.closed or n < 16:
        return np.zeros(len(tr.s))
    st = max(1, int(round(RL_KNOT / tr.ds)), int(math.ceil(n / RL_KNOTS_MAX)))
    ic = np.arange(0, n, st)
    m = len(ic)
    c = tr.xy[ic]
    nu = _nrm(tr)[ic]
    D = (np.roll(np.eye(m), -1, axis=1) - 2.0 * np.eye(m)
         + np.roll(np.eye(m), 1, axis=1))
    A = np.vstack([D * nu[None, :, 0], D * nu[None, :, 1]])
    b = np.concatenate([D @ c[:, 0], D @ c[:, 1]])
    H = A.T @ A + 1e-6 * np.eye(m)             # a straight is otherwise free
    g = A.T @ b
    lim = max(hw - RL_MARGIN, 0.0)
    lo, hi = -lim, lim
    x = np.zeros(m)
    fixed = np.zeros(m, dtype=bool)
    val = np.zeros(m)
    for _ in range(80):
        free = ~fixed
        if free.any():
            rhs = -(g[free] + H[np.ix_(free, fixed)] @ val[fixed])
            x[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
        x[fixed] = val[fixed]
        vh, vl = free & (x > hi + 1e-9), free & (x < lo - 1e-9)
        if vh.any() or vl.any():
            fixed |= vh | vl
            val[vh], val[vl] = hi, lo
            continue
        grad = H @ x + g
        rel = fixed & (((val >= hi) & (grad > 1e-9)) | ((val <= lo) & (grad < -1e-9)))
        if not rel.any():
            break
        fixed &= ~rel
    x = np.clip(x, lo, hi)
    # knots -> every sample (periodic), then a light smooth
    sk = ic * tr.ds
    L = n * tr.ds
    s_all = np.arange(n) * tr.ds
    xs = np.interp(s_all, np.concatenate([sk, [L]]), np.concatenate([x, [x[0]]]))
    w = 9
    ker = np.ones(w) / w
    xs = np.convolve(np.concatenate([xs[-w:], xs, xs[:w]]), ker, 'same')[w:-w]
    out = np.empty(len(tr.s))
    out[:n] = xs
    out[n:] = xs[0]
    return np.clip(out, lo, hi)


# ======================================================================= #
#  the layout                                                             #
# ======================================================================= #
class Layout:
    """Everything beside and on the road for one track, in world metres.

    Built once per track by `layout(tr)` (cached by identity). Arrays:
      kerb_s0/s1/side/col/kind   kerb blocks (kind 0 inside, 1 exit)
      kerbL, kerbR               per-sample bool: a kerb block covers it
      kerbL0, kerbR0             ... render's inside-kerb rule alone (what is
                                 drawn with scenery=False)
      grav_lo/hi (2,N)           per-sample gravel n-range, [0] left [1] right
                                 (|n|, 0 = none) -- wheel_surfaces reads this
      chunks                     ground polygons: list of dict(poly, kind,
                                 c (centre), r (radius)) kind 'gravel' | 'runoff'
      runoff_paint               (K,4,2) painted bands on the run-off
      rakes                      (K,2,2) rake lines in the gravel
      n_rl                       (N,) racing-line offset (None: no rubber)
      patches                    (K,4,2) + patch_col (K,) asphalt repairs
      cracks                     (K,2,2) sealed-crack segments + crack_s (K,)
      paint / paint_col / paint_s  (K,4,2) static paint quads (grid boxes,
                                 launch box, distance numerals, gate bars),
                                 corners at EXACT s (frame_at)
      grid_s                     grid rows' arclengths, where the cars' CGs
                                 stand (None: no grid); grid_slots (K,2) the
                                 (s, n) of every painted box's car
      lines                      long painted lines: list of dict(n, w, key,
                                 kerb_break, s_lo, s_hi)
      guide_rings                list of (cx, cy, R, on_ribbon) (skidpad)
      stripe_dir                 unit vector ALONG the mowing stripes
      dashes                     bool: the centre dashes stay (open map)
      verge                      bool: the verge band is drawn along the road
      end_trap                   None or dict (dragstrip sand trap)
    """

    def __init__(self, tr):
        t0 = time.perf_counter()
        self.name = str(tr.name)
        self.theme = theme_of(tr)
        self.closed = bool(tr.closed)
        n = _nper(tr)
        N = len(tr.s)
        self.N = N
        self.hw = hw = 0.5 * float(tr.width)
        self.L = float(tr.length)
        nrm = _nrm(tr)
        self.corners = corners(tr)
        theme = self.theme
        rng = rng_for(tr)

        # ---------------- kerbs -------------------------------------------
        blocks = {}                     # (k, side) -> kind
        nb = int(math.ceil(self.L / KERB_BLOCK)) if tr.closed else int(
            math.ceil(self.L / KERB_BLOCK))
        kab = np.abs(tr.kappa)
        for kb in range(nb):
            s = kb * KERB_BLOCK
            i0 = int(_i_of_s(tr, s))
            if kab[i0] > KERB_K_FULL:                  # render's rule, verbatim
                blocks[(kb, 1 if tr.kappa[i0] > 0 else -1)] = 0
        # render's own rule alone: what is drawn with scenery=False, which
        # wheel_surfaces must then report (no apex / exit kerbs on screen)
        rule_keys = sorted(blocks)

        def _add(sa, sb, side, kind):
            for kb in range(int(math.floor(sa / KERB_BLOCK)),
                            int(math.ceil(sb / KERB_BLOCK))):
                if tr.closed:
                    kb %= nb
                elif kb < 0 or kb >= nb:
                    continue
                blocks.setdefault((kb, side), kind)

        if theme in ("circuit", "open"):
            for c in self.corners:
                ln = c["s1"] - c["s0"]
                if c["R"] >= KERB_R_APEX:
                    continue
                if c["R"] > 1.0 / KERB_K_FULL:         # apex kerb, inside
                    mid = 0.5 * (c["s0"] + c["s1"])
                    _add(mid - APEX_FRAC * ln, mid + APEX_FRAC * ln, c["sgn"], 0)
                back = min(0.30 * ln, EXIT_KERB_BACK)
                _add(c["s1"] - back, c["s1"] + EXIT_KERB_ON, -c["sgn"], 1)
        keys = sorted(blocks)
        self.kerb_s0 = np.array([k * KERB_BLOCK for k, _ in keys], dtype=np.float64)
        self.kerb_s1 = np.minimum(self.kerb_s0 + KERB_BLOCK, self.L)
        self.kerb_side = np.array([sd for _, sd in keys], dtype=np.int8)
        self.kerb_col = np.array([k % 2 for k, _ in keys], dtype=np.int8)
        self.kerb_kind = np.array([blocks[k] for k in keys], dtype=np.int8)
        def _tables(keys_):
            kl, kr = np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)
            for kb_, sd in keys_:
                s0 = kb_ * KERB_BLOCK
                i0, i1 = int(_i_of_s(tr, s0)), int(_i_of_s(tr, min(s0 + KERB_BLOCK, self.L)))
                idx = (np.arange(i0, i0 + ((i1 - i0) % n if tr.closed else i1 - i0) + 1)
                       % n) if tr.closed else np.arange(i0, i1 + 1)
                (kl if sd > 0 else kr)[idx] = True
            if tr.closed:
                kl[n:], kr[n:] = kl[0], kr[0]
            return kl, kr
        self.kerbL, self.kerbR = _tables(keys)
        self.kerbL0, self.kerbR0 = _tables(rule_keys)

        # ---------------- gravel traps / run-off --------------------------
        self.grav_lo = np.zeros((2, N))
        self.grav_hi = np.zeros((2, N))
        self.chunks = []
        paint, rakes = [], []
        self._edges = []
        if theme == "circuit":
            for c in self.corners:
                if c["R"] >= CORNER_R_MAX * 0.75:
                    continue
                side = -c["sgn"]                       # the OUTSIDE
                if c["R"] < GRAVEL_R_MAX:
                    kind = "gravel"
                    sa, sb = c["s0"] - GRAVEL_LEAD, c["s1"] + GRAVEL_OVERRUN
                    n_in = hw + VERGE_W
                    depth = min(DRESS_MAX - VERGE_W, 10.0 + 0.20 * c["R"])
                else:
                    kind = "runoff"
                    sa, sb = c["s0"] - RUNOFF_LEAD, c["s1"] + RUNOFF_OVERRUN
                    n_in = hw
                    depth = min(DRESS_MAX, 8.0 + 0.03 * c["R"])
                self._trap(tr, nrm, kind, sa, sb, side, n_in, depth, paint, rakes)
        self.runoff_paint = (np.array(paint, dtype=np.float64).reshape(-1, 4, 2))
        # the traps' borders (gravel meets grass / verge): drawn as a line, so
        # a trap has a defined edge rather than a colour step
        self.trap_edges = np.array(self._edges, dtype=np.float64).reshape(-1, 2, 2)
        self.rakes = np.array(rakes, dtype=np.float64).reshape(-1, 2, 2)

        # ---------------- the dragstrip's sand trap past the end ----------
        self.end_trap = None
        if theme == "dragstrip":
            e = tr.xy[-1]
            f = np.array([math.cos(tr.psi[-1]), math.sin(tr.psi[-1])])
            lt = np.array([-f[1], f[0]])
            half = hw + VERGE_W + 2.0
            poly = np.array([e + 1.0 * f + half * lt, e + TRAP_END * f + half * lt,
                             e + TRAP_END * f - half * lt, e + 1.0 * f - half * lt])
            self.end_trap = dict(e=e, f=f, lt=lt, half=half, d0=1.0, d1=TRAP_END)
            self.chunks.append(dict(poly=poly, kind="gravel", c=poly.mean(axis=0),
                                    r=float(np.hypot(*(poly - poly.mean(axis=0)).T).max())))
            for fr in (0.3, 0.55, 0.8):
                rakes.append((e + (1.0 + 0.0) * f + (fr * 2 - 1) * half * lt,
                              e + TRAP_END * f + (fr * 2 - 1) * half * lt))
            self.rakes = np.array(rakes, dtype=np.float64).reshape(-1, 2, 2)
        if self.chunks:
            self.chunk_c = np.array([ch["c"] for ch in self.chunks])
            self.chunk_r = np.array([ch["r"] for ch in self.chunks])
        else:
            self.chunk_c = np.zeros((0, 2))
            self.chunk_r = np.zeros(0)

        # ---------------- the rubbered line --------------------------------
        self.n_rl = None
        if theme in ("circuit", "open"):
            self.n_rl = _racing_line_cached(tr, hw)
        elif theme == "skidpad":
            self.n_rl = np.zeros(N)                   # the driven radius

        # ---------------- repairs and cracks (ON the ribbon) ---------------
        pq, pc = [], []
        cr, crs = [], []
        if theme in ("circuit", "open"):
            wet = [(p.s0, p.s1 if p.s1 >= p.s0 else p.s1 + self.L)
                   for p in tr.surfaces]
            npatch = max(2, int(self.L / PATCH_EVERY))
            tries = 0
            while len(pq) < npatch and tries < 200:
                tries += 1
                s0 = float(rng.uniform(40.0, self.L - 40.0))
                ln = float(rng.uniform(3.5, 11.0))
                wd = float(rng.uniform(1.6, 5.0))
                nc = float(rng.uniform(-hw + 0.6 + 0.5 * wd, hw - 0.6 - 0.5 * wd))
                if any(a - 5.0 < s0 + ln and s0 < b + 5.0 for a, b in wet):
                    continue
                ang = float(rng.uniform(-0.08, 0.08))          # not quite square
                i0 = int(_i_of_s(tr, s0))
                i1 = int(_i_of_s(tr, s0 + ln))
                a0, a1 = nc - 0.5 * wd, nc + 0.5 * wd
                pq.append([tr.xy[i0] + (a0 + ang * wd) * nrm[i0],
                           tr.xy[i0] + (a1 + ang * wd) * nrm[i0],
                           tr.xy[i1] + a1 * nrm[i1], tr.xy[i1] + a0 * nrm[i1]])
                pc.append(int(rng.integers(0, 2)))
            s = float(rng.uniform(*CRACK_EVERY))
            while s < self.L - 5.0:
                span = float(rng.uniform(0.15, 0.55)) * 2.0 * hw
                npts = max(4, int(span / 0.7))
                n0 = float(rng.uniform(-hw + 0.3, hw - 0.3 - span))
                ns = n0 + np.linspace(0.0, span, npts) + rng.normal(0.0, 0.12, npts)
                ss = (s + np.cumsum(rng.normal(0.0, 0.22, npts))
                      + np.linspace(0, float(rng.uniform(-1.2, 1.2)), npts))
                ii = _i_of_s(tr, ss)
                P = tr.xy[ii] + ns[:, None] * nrm[ii]
                for a in range(npts - 1):
                    cr.append((P[a], P[a + 1]))
                    crs.append(s)
                s += float(rng.uniform(*CRACK_EVERY))
        self.patches = np.array(pq, dtype=np.float64).reshape(-1, 4, 2)
        self.patch_col = np.array(pc, dtype=np.int8)
        self.cracks = np.array(cr, dtype=np.float64).reshape(-1, 2, 2)
        self.crack_s = np.array(crs, dtype=np.float64)

        # ---------------- static paint quads -------------------------------
        pq2, pcol, ps = [], [], []

        def quad(s0, s1, n0, n1, key):
            # exact-s corners (frame_at): a 0.2 m bar is 0.2 m, not 0 or 0.5
            pq2.append(quads_at(tr, s0, s1, n0, n1, nrm)[0])
            pcol.append(key)
            ps.append(0.5 * (s0 + s1))

        self.grid_s = None
        self.grid_slots = np.zeros((0, 2))
        if theme == "circuit" and tr.closed:
            # grid boxes WHERE THE RACE LINES CARS UP: rows GRID_ROW_M apart
            # back from the line (s = 0, -7, -14 ...), two a row at
            # +-GRID_SIDE_N and your car on n = 0 in the front row, each box
            # following the road's curve (exact-s corners) -- the arena's
            # line is the end of T7, R 130 m, so its grid stands in that
            # gentle arc as its cars do. A box is a bar across the road just
            # ahead of the car's nose and two ticks back from it outside the
            # car's flanks; a row is painted only while every metre from
            # the rearmost tail to the bar is gentler than GRID_R_MIN.
            # (Eight boxes on the straight AFTER the line, s 7-63 m, left
            # every car at a race start standing behind empty slots.)
            kab = np.abs(tr.kappa[:n])
            s_line = float(tr.sector_s[0]) if len(tr.sector_s) else 0.0
            bar0, bar1 = GRID_NOSE + 0.10, GRID_NOSE + 0.30       # m past the CG
            tick0 = bar0 - 1.40
            slots, rows = [], []
            for k in range(GRID_ROWS):
                sc_ = s_line - GRID_ROW_M * k
                ss_ = np.arange(sc_ - GRID_TAIL - 0.3, sc_ + bar1 + 0.01, 0.25)
                if kab[_i_of_s(tr, ss_)].max() > 1.0 / GRID_R_MIN:
                    break
                ns_ = (GRID_SIDE_N, 0.0, -GRID_SIDE_N) if k == 0 else (GRID_SIDE_N, -GRID_SIDE_N)
                ns_ = [nc for nc in ns_ if abs(nc) + GRID_BOX_HALF <= hw - EDGE_INSET - EDGE_LINE_W]
                if not ns_:
                    break
                rows.append(sc_ % self.L)
                for nc in ns_:
                    slots.append((sc_ % self.L, nc))
                    a0, a1 = nc - GRID_BOX_HALF, nc + GRID_BOX_HALF
                    quad(sc_ + bar0, sc_ + bar1, a0, a1, PAINT_WHITE)
                    # the ticks stop short of the chequer (the front row's
                    # would run 0.75 m over its squares)
                    t0_ = sc_ + tick0
                    if t0_ < s_line + CHEQUER_L and sc_ + bar0 > s_line:
                        t0_ = s_line + CHEQUER_L
                    quad(t0_, sc_ + bar0, a0, a0 + 0.14, PAINT_WHITE)
                    quad(t0_, sc_ + bar0, a1 - 0.14, a1, PAINT_WHITE)
            self.grid_s = np.array(rows, dtype=np.float64)
            self.grid_slots = np.array(slots, dtype=np.float64).reshape(-1, 2)
        if theme == "dragstrip":
            lh = DRAG_LANE_HALF
            # the launch box: heavy rubber inside, a bar across its far end
            quad(0.0, DRAG_LAUNCH_BOX, -lh + 0.2, lh - 0.2, PAINT_RUBBER)
            quad(DRAG_LAUNCH_BOX, DRAG_LAUNCH_BOX + 0.25, -lh, lh, PAINT_WHITE)
            # distance paint: a bar across the lane and the hundreds of
            # metres in 7-segment numerals on both shoulders, readable from
            # the car that is driving toward them
            for k in range(1, int(self.L // 100.0) + 1):
                sk = 100.0 * k
                if sk + 0.3 > self.L - 0.5:
                    break                     # the strip's END, not a distance
                quad(sk, sk + 0.3, -lh, lh, PAINT_DIM)
                txt = str(k)
                for side in (1.0, -1.0):
                    n_left = (lh + 0.8 + 3.5) if side > 0 else (-lh - 0.8)
                    for q in _numeral_quads(txt, s_base=sk - 4.2, n_left=n_left,
                                            height=3.2, width=1.5, stroke=0.32,
                                            gap=0.45):
                        quad(*q, PAINT_DIM)
            for s_g, _lbl in getattr(tr, "gates", []):
                if s_g > 0.5:
                    quad(s_g, s_g + 0.35, -hw + 0.4, hw - 0.4, PAINT_WHITE)
        self.paint = np.array(pq2, dtype=np.float64).reshape(-1, 4, 2)
        self.paint_col = np.array(pcol, dtype=np.int8)
        self.paint_s = np.array(ps, dtype=np.float64)

        # ---------------- long painted lines --------------------------------
        e = hw - EDGE_INSET
        self.lines = [dict(n=+e, w=EDGE_LINE_W, key=PAINT_WHITE, kerb_break=+1),
                      dict(n=-e, w=EDGE_LINE_W, key=PAINT_WHITE, kerb_break=-1)]
        self.dashes = theme == "open"
        if theme == "skidpad":
            self.lines.append(dict(n=0.0, w=0.15, key=PAINT_WHITE, kerb_break=0))
        if theme == "dragstrip":
            for sgn in (1.0, -1.0):
                self.lines.append(dict(n=sgn * DRAG_LANE_HALF, w=0.12,
                                       key=PAINT_WHITE, kerb_break=0))
        self.verge = theme != "open"
        # the drag groove: the wheel tracks rubbered, darkest at the launch
        self.grooves = []
        if theme == "dragstrip":
            for s0, s1, shade in DRAG_GROOVE:
                for ny in (0.7145, -0.7145):
                    self.grooves.append((s0, s1, ny, 0.24, shade))

        # ---------------- skidpad guide circles -----------------------------
        self.guide_rings = []
        cen = getattr(tr, "centre", ()) or ()
        if theme == "skidpad" and len(cen) == 2:
            R_drive = float(np.hypot(*(tr.xy[0] - np.asarray(cen))))
            for R in getattr(tr, "guide_radii", []):
                on_rib = abs(float(R) - R_drive) <= hw
                if abs(float(R) - R_drive) < 0.5:
                    continue                      # the centre line already
                self.guide_rings.append((float(cen[0]), float(cen[1]), float(R), on_rib))

        # ---------------- mowing stripes -------------------------------------
        if theme == "dragstrip" or not tr.closed:
            ang = float(tr.psi[0])
        else:
            # along the longest straight: circuits mow parallel to the main
            # straight, so the bands run with the road where it matters most
            straight = np.abs(tr.kappa[:n]) < 1e-4
            best, run, best_i, a = 0, 0, 0, 0
            for i in range(n):
                if straight[i]:
                    if run == 0:
                        a = i
                    run += 1
                    if run > best:
                        best, best_i = run, a
                else:
                    run = 0
            ang = float(tr.psi[best_i + best // 2]) if best else 0.0
        if theme == "skidpad":
            ang = 0.35                             # a diagonal against the ring
        self.stripe_dir = np.array([math.cos(ang), math.sin(ang)])
        self.build_ms = (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------ #
    def _trap(self, tr, nrm, kind, sa, sb, side, n_in, depth, paint, rakes):
        """One gravel trap or run-off outside a corner: per-sample n-range,
        trimmed where another stretch of road (or a drivable area) is
        nearer, then cut into CHUNK_M chunks for drawing."""
        n = _nper(tr)
        ds = float(tr.ds)
        ss = np.arange(sa, sb + 0.5 * ds, ds)
        if not tr.closed:
            ss = ss[(ss >= 0.0) & (ss <= self.L)]
        if len(ss) < 8:
            return
        ln = max(ss[-1] - ss[0], 1e-9)
        # a rounded lens that fills out over the first 40 % and tapers over
        # the last 30 %: fullest late, because cars leave the road late in a
        # corner, not at its entry; smoothstep ends, so no trap starts with
        # a cliff
        def smooth(x):
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)
        prof = smooth((ss - ss[0]) / (0.40 * ln)) * smooth((ss[-1] - ss) / (0.30 * ln))
        prof = np.sqrt(prof)
        n_out = n_in + depth * prof
        ii = _i_of_s(tr, ss)
        # trim: the outer boundary must still project onto THIS stretch of
        # road, i.e. nothing else is nearer. Bisection on the depth.
        for j in range(0, len(ss), 2):
            lo_d, hi_d = 0.0, n_out[j] - n_in
            ok_full = self._owns(tr, nrm, ii[j], side, n_in + hi_d)
            if ok_full:
                continue
            for _ in range(7):
                mid = 0.5 * (lo_d + hi_d)
                if self._owns(tr, nrm, ii[j], side, n_in + mid):
                    lo_d = mid
                else:
                    hi_d = mid
            n_out[j] = n_in + lo_d
            if j + 1 < len(ss):
                n_out[j + 1] = min(n_out[j + 1], n_out[j])
        # a trimmed sample must not leave a spike: running min over +-4 m
        w = int(round(4.0 / ds))
        pad = np.concatenate([np.full(w, n_in), n_out, np.full(w, n_in)])
        n_out = np.min(np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1), axis=1)
        n_out = np.maximum(n_out, n_in)
        keep = (n_out - n_in) > 0.5
        if keep.sum() < 8:
            return
        # the per-sample table wheel_surfaces reads
        row = 0 if side > 0 else 1
        lo_t, hi_t = self.grav_lo[row], self.grav_hi[row]
        if kind == "gravel":
            for j in np.flatnonzero(keep):
                i = int(ii[j])
                lo_t[i], hi_t[i] = n_in, max(hi_t[i], n_out[j])
        # chunks for the drawing, every 4th sample (2 m chords: 1 cm sagitta
        # on the outside of the R = 30 m hairpin)
        step = max(1, int(round(2.0 / ds)))
        js = np.flatnonzero(keep)
        per = max(2, int(round(CHUNK_M / (step * ds))))
        pts_j = js[::step]
        if pts_j[-1] != js[-1]:
            pts_j = np.append(pts_j, js[-1])
        if kind == "gravel":
            i_e = ii[pts_j]
            ob = tr.xy[i_e] + side * n_out[pts_j][:, None] * nrm[i_e]
            ib = tr.xy[i_e] + side * n_in * nrm[i_e]
            for P in (ob, ib):
                for a in range(len(P) - 1):
                    self._edges.append((P[a], P[a + 1]))
        for a in range(0, len(pts_j) - 1, per):
            sel = pts_j[a:a + per + 1]
            if len(sel) < 2:
                continue
            i_s = ii[sel]
            inner = tr.xy[i_s] + side * n_in * nrm[i_s]
            outer = tr.xy[i_s] + side * n_out[sel][:, None] * nrm[i_s]
            poly = np.vstack([inner, outer[::-1]])
            cc = poly.mean(axis=0)
            self.chunks.append(dict(poly=poly, kind=kind, c=cc,
                                    r=float(np.hypot(*(poly - cc).T).max())))
        if kind == "runoff":
            # abrasive paint: bands across the run-off every 6 m, slanted
            # with the flow of the corner
            w3, sl = int(round(3.8 / ds)), int(round(2.5 / ds))
            for j0 in js[:: int(round(6.0 / ds))]:
                j1, k0, k1 = j0 + w3, j0 + sl, j0 + w3 + sl
                if k1 >= len(ss) or not (keep[j0] and keep[k1]):
                    continue
                a_out = min(n_out[j0], n_out[j1], n_out[k0], n_out[k1]) - 0.25
                if a_out - n_in < 1.5:
                    continue
                q = [(j0, n_in + 0.9), (j1, n_in + 0.9), (k1, a_out), (k0, a_out)]
                paint.append([tr.xy[ii[j]] + side * nn * nrm[ii[j]] for j, nn in q])
        else:
            for fr in (0.28, 0.52, 0.76):
                seg = js[::step]
                P = tr.xy[ii[seg]] + side * (n_in + fr * (n_out[seg] - n_in))[:, None] * nrm[ii[seg]]
                live = (n_out[seg] - n_in) > 2.0
                for a in range(len(seg) - 1):
                    if live[a] and live[a + 1]:
                        rakes.append((P[a], P[a + 1]))

    def _owns(self, tr, nrm, i, side, nn) -> bool:
        """Does the point `nn` metres to `side` of sample i still project
        onto sample i's own stretch of road (and lie on no drivable area)?"""
        p = tr.xy[i] + side * nn * nrm[i]
        s_p, n_p, _, _, _ = trk.project(tr, float(p[0]), float(p[1]))
        ds_ = abs(s_p - tr.s[i])
        if tr.closed:
            ds_ = min(ds_, self.L - ds_)
        if ds_ > 4.0 or abs(n_p - side * nn) > 0.6:
            return False
        for ar in getattr(tr, "areas", []):
            if ar.drivable and ar.contains(float(p[0]), float(p[1])):
                return False
        return True


def _numeral_quads(txt, s_base, n_left, height, width, stroke, gap):
    """7-segment numerals painted flat on the road, read from a car driving
    toward +s: the glyph's up is +s, its right is -n. Quads as (s0, s1, n0,
    n1) rectangles, which is all a segment is."""
    SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
           "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg"}
    out = []
    h, w, t = height, width, stroke
    for j, ch in enumerate(txt):
        nl = n_left - j * (w + gap)           # glyph's left edge (larger n)
        nr = nl - w
        s0, sm, s1 = s_base, s_base + 0.5 * h, s_base + h
        rects = {"a": (s1 - t, s1, nr, nl), "d": (s0, s0 + t, nr, nl),
                 "g": (sm - 0.5 * t, sm + 0.5 * t, nr, nl),
                 "f": (sm, s1, nl - t, nl), "e": (s0, sm, nl - t, nl),
                 "b": (sm, s1, nr, nr + t), "c": (s0, sm, nr, nr + t)}
        for sg in SEG.get(ch, ""):
            out.append(rects[sg])
    return out


# --- caches: per track object (the renderer and wheel_surfaces share one),
#     and the racing line per GEOMETRY, because every self-check builds a
#     fresh Track of the same map ------------------------------------------
_LAYOUTS: dict = {}
_RL_CACHE: dict = {}


def _geom_key(tr):
    return (str(tr.name), round(float(tr.length), 4), len(tr.s),
            round(float(tr.width), 3), round(float(tr.xy[0][0]), 3),
            round(float(tr.xy[0][1]), 3), round(float(tr.psi[0]), 4))


def _racing_line_cached(tr, hw):
    k = _geom_key(tr)
    v = _RL_CACHE.get(k)
    if v is None:
        v = _racing_line(tr, hw)
        _RL_CACHE[k] = v
    return v


def layout(tr) -> Layout:
    """The Layout of `tr`, built once per track object."""
    ent = _LAYOUTS.get(id(tr))
    if ent is not None and ent[0]() is tr:
        return ent[1]
    lay = Layout(tr)
    if len(_LAYOUTS) > 32:
        for k in [k for k, (r, _l) in _LAYOUTS.items() if r() is None]:
            del _LAYOUTS[k]
    _LAYOUTS[id(tr)] = (weakref.ref(tr), lay)
    return lay


# ======================================================================= #
#  what each wheel is on                                                   #
# ======================================================================= #
def surface_at(tr, x: float, y: float, on=None, mu=None, scenery: bool = True) -> str:
    """One point: 'tarmac' | 'wet' | 'kerb' | 'grass' | 'gravel'. `on` / `mu`
    are the physics' own answer for that point when the caller has it.
    scenery=False answers for what render draws WITHOUT the world: its own
    inside-kerb rule (|kappa| > 1/60) and no traps -- grass off the road."""
    lay = layout(tr)
    s, n, _k, _p, i = trk.project(tr, float(x), float(y))
    if on is None:
        on = trk.on_tarmac(tr, float(x), float(y), n)
    if on:
        if mu is not None and float(mu) < 0.95:
            return "wet"
        hw = lay.hw
        if abs(n) <= hw and abs(n) >= hw - KERB_W:
            if scenery:
                tab = lay.kerbL if n > 0 else lay.kerbR
            else:
                tab = lay.kerbL0 if n > 0 else lay.kerbR0
            if tab[i]:
                return "kerb"
        return "tarmac"
    if not scenery:
        return "grass"
    et = lay.end_trap
    if et is not None:
        d = np.array([x, y]) - et["e"]
        a, b = float(d @ et["f"]), float(d @ et["lt"])
        # from d0: the trap is DRAWN from 1 m past the end (grass before it)
        if et["d0"] <= a <= et["d1"] and abs(b) <= et["half"]:
            return "gravel"
    row = 0 if n > 0 else 1
    an = abs(n)
    lo, hi = lay.grav_lo[row][i], lay.grav_hi[row][i]
    if hi > 0.0 and lo <= an <= hi:
        return "gravel"
    return "grass"


def wheel_surfaces(track, pts, on_track4=None, mu4=None, scenery: bool = True) -> tuple:
    """Per wheel (FL FR RL RR): 'tarmac' | 'wet' | 'kerb' | 'grass' | 'gravel'.

    pts: the four contact points in world metres. on_track4 / mu4: what the
    physics' own surface lookup said for each wheel (Sim.on_track4, Sim.mu),
    when the caller has them. Agrees with the drawing by construction: it
    projects each point onto the centreline and reads the same per-sample
    tables the renderer offsets to draw the kerbs and the traps. Measured
    ~35 us for four wheels (four `track.project` calls, ~7 us each).

    scenery: pass the renderer's ViewConfig.scenery. With the world off the
    screen shows only render's own inside kerbs and no traps, so the sound
    and the particles must not report an apex kerb or gravel nobody can see.
    The grey run-off beside the fast corners reports 'grass': the physics
    gives it the off-track grip (track.MU_OFF_TRACK), whatever is painted.
    """
    if pts is None:
        return tuple("tarmac" for _ in range(4))
    out = []
    for w in range(4):
        on = None if on_track4 is None else bool(on_track4[w])
        mu = None if mu4 is None else float(mu4[w])
        if on and mu is not None and mu < 0.95:
            out.append("wet")           # no projection needed
            continue
        out.append(surface_at(track, float(pts[w][0]), float(pts[w][1]), on, mu, scenery))
    return tuple(out)


# ======================================================================= #
#  self-check                                                              #
# ======================================================================= #
def self_check(verbose: bool = True) -> bool:
    ok_all = True
    res = []

    def rep(tag, passed, msg):
        nonlocal ok_all
        ok_all = ok_all and bool(passed)
        res.append((tag, passed, msg))
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {tag:44s} {msg}")

    if verbose:
        print("drive/scenery.py self-check")
    tracks = {nm: trk.make_track(nm) for nm in trk.TRACK_ORDER}
    lays = {nm: layout(t) for nm, t in tracks.items()}
    ar, la = tracks["arena"], lays["arena"]

    # determinism: a fresh Track of the same map lays out bit-identically
    t2 = trk.make_track("arena")
    lb = Layout(t2)
    same = (np.array_equal(la.kerb_s0, lb.kerb_s0) and np.array_equal(la.patches, lb.patches)
            and np.array_equal(la.cracks, lb.cracks) and len(la.chunks) == len(lb.chunks)
            and all(np.array_equal(a["poly"], b["poly"]) for a, b in zip(la.chunks, lb.chunks))
            and np.array_equal(la.n_rl, lb.n_rl) and np.array_equal(la.paint, lb.paint))
    rep("deterministic (seeded from the track name)", same,
        f"{len(la.kerb_s0)} kerb blocks, {len(la.chunks)} trap chunks, "
        f"{len(la.patches)} patches, {len(la.cracks)} crack segs, twice")
    c = corners(ar)
    radii = sorted(round(x["R"]) for x in c)
    rep("arena: seven corners found from the curvature",
        len(c) == 7 and radii == [30, 35, 45, 55, 80, 110, 130], f"R = {radii}")
    nk = {k: int((la.chunks and sum(1 for ch in la.chunks if ch['kind'] == k)) or 0)
          for k in ("gravel", "runoff")}
    rep("arena: gravel outside the slow corners, run-off outside the fast",
        nk["gravel"] > 0 and nk["runoff"] > 0,
        f"{nk['gravel']} gravel chunks (R < {GRAVEL_R_MAX:.0f}), {nk['runoff']} run-off chunks")

    # nothing of the dressing lies on the road; traps within DRESS_MAX
    worst_on, worst_far = 1e9, 0.0
    for nm, lay in lays.items():
        t = tracks[nm]
        hw = lay.hw
        for ch in lay.chunks:
            P = ch["poly"]
            dense = np.vstack([P[a] + f * (P[(a + 1) % len(P)] - P[a])
                               for a in range(len(P)) for f in (0.0, 0.5)])
            for p in dense:
                if nm == "dragstrip":
                    continue                 # past the END: n is meaningless there
                s_p, n_p, _, _, _ = trk.project(t, float(p[0]), float(p[1]))
                # a run-off's inner boundary IS the edge (0.00); a vertex on
                # a drivable Area counts as on the road
                in_area = any(a_.drivable and a_.contains(float(p[0]), float(p[1]))
                              for a_ in getattr(t, "areas", []))
                worst_on = min(worst_on, -1.0 if in_area else abs(n_p) - hw)
                worst_far = max(worst_far, abs(n_p) - hw)
    rep("no trap / run-off vertex on the road", worst_on >= -0.05,
        f"closest vertex {worst_on:+.2f} m from the edge (must be >= 0)")
    rep(f"every trap / run-off within {DRESS_MAX:.0f} m of the edge",
        worst_far <= DRESS_MAX + 0.05, f"farthest vertex {worst_far:.2f} m out")
    ok_pq = True
    for p4 in la.patches.reshape(-1, 2):
        s_p, n_p, _, _, _ = trk.project(ar, float(p4[0]), float(p4[1]))
        ok_pq &= abs(n_p) <= la.hw + 0.1
    rep("repair patches lie on the ribbon", ok_pq and len(la.patches) >= 2,
        f"{len(la.patches)} patches, all |n| <= hw")
    rl = la.n_rl
    kap = ar.kappa[:len(rl)]
    apex = [float(np.sign(np.mean(rl[x["i0"] + x["n"] // 3: x["i0"] + 2 * x["n"] // 3]))
                  == x["sgn"]) for x in c if x["R"] < 100.0]
    rep("racing line: inside at the apexes, inside the edges",
        np.abs(rl).max() <= la.hw - RL_MARGIN + 1e-9 and all(apex),
        f"max |n| {np.abs(rl).max():.2f} m (<= {la.hw - RL_MARGIN:.2f}), "
        f"apex side right in {int(sum(apex))}/{len(apex)} corners (R < 100)")
    _ = kap

    # wheel_surfaces on known points
    def pt(t, s, n):
        return trk.point_at(t, s, n)
    got = {}
    # a kerb block: inside of T2 (R = 30, left-hander), block middle
    j = int(np.flatnonzero((la.kerb_side > 0) & (la.kerb_kind == 0)
                           & (la.kerb_s0 > 300.0))[0])
    sk = 0.5 * (la.kerb_s0[j] + la.kerb_s1[j])
    got["kerb"] = surface_at(ar, *pt(ar, sk, la.hw - 0.4))
    # a gravel trap: the middle of the deepest gravel sample
    row = 1 if la.grav_hi[1].max() >= la.grav_hi[0].max() else 0
    ig = int(np.argmax(la.grav_hi[row]))
    sg = 1.0 if row == 0 else -1.0
    got["gravel"] = surface_at(ar, *pt(ar, ar.s[ig], sg * 0.5 * (la.grav_lo[row][ig] + la.grav_hi[row][ig])))
    got["tarmac"] = surface_at(ar, *pt(ar, 60.0, 0.0))
    got["grass"] = surface_at(ar, *pt(ar, 60.0, la.hw + 5.0))
    op = tracks["open"]
    got["open pad"] = surface_at(op, 105.0, 150.0)
    iw = int(round(500.0 / ar.ds))
    mu_w = trk.surface_at(ar, *pt(ar, 500.0, 0.0))[0]
    got["wet"] = wheel_surfaces(ar, [pt(ar, 500.0, 0.0)] * 4, [True] * 4, [mu_w] * 4)[0]
    got["drag trap"] = surface_at(tracks["dragstrip"], 1500.0 + 20.0 + 10.0, 40.0)
    want = {"kerb": "kerb", "gravel": "gravel", "tarmac": "tarmac", "grass": "grass",
            "open pad": "tarmac", "wet": "wet", "drag trap": "gravel"}
    bad = {k: v for k, v in got.items() if v != want[k]}
    rep("wheel_surfaces on known points", not bad,
        ", ".join(f"{k}->{v}" for k, v in got.items()) + (f"  WRONG {bad}" if bad else ""))
    _ = iw
    # the drag trap starts where it is DRAWN (d0 = 1 m past the end), and
    # scenery=False answers for render's own look: inside kerbs only, no traps
    dr = tracks["dragstrip"]
    e_ = lays["dragstrip"].end_trap
    near_end = e_["e"] + 0.5 * e_["f"]
    s_ne = surface_at(dr, float(near_end[0]), float(near_end[1]))
    jx = np.flatnonzero((la.kerb_kind == 1))[0]            # an exit kerb block
    sx_ = 0.5 * (la.kerb_s0[jx] + la.kerb_s1[jx])
    px_ = pt(ar, sx_, float(la.kerb_side[jx]) * (la.hw - 0.4))
    on_rule = bool((la.kerbL0 if la.kerb_side[jx] > 0 else la.kerbR0)[int(_i_of_s(ar, sx_))])
    s_on, s_off = surface_at(ar, *px_), surface_at(ar, *px_, scenery=False)
    pg_ = pt(ar, ar.s[ig], sg * 0.5 * (la.grav_lo[row][ig] + la.grav_hi[row][ig]))
    g_off = wheel_surfaces(ar, [pg_] * 4, [False] * 4, [0.55] * 4, scenery=False)[0]
    k_in = surface_at(ar, *pt(ar, sk, la.hw - 0.4), scenery=False)
    rep("drag trap from d0; scenery=False -> render's own kerbs, no traps",
        s_ne == "grass" and s_on == "kerb" and (on_rule or s_off == "tarmac")
        and g_off == "grass" and k_in == "kerb",
        f"0.5 m past the strip {s_ne}; exit kerb {s_on} / off {s_off}; "
        f"trap off {g_off}; inside kerb off {k_in}")
    # paint quads keep their length along the road (exact-s corners)
    lens = []
    for nm in ("arena", "dragstrip"):
        P = lays[nm].paint
        m0, m1 = 0.5 * (P[:, 0] + P[:, 1]), 0.5 * (P[:, 2] + P[:, 3])
        lens.append(float(np.hypot(*(m1 - m0).T).min()))
    rep("paint quads from exact s: none collapses", min(lens) > 0.15,
        f"shortest along the road: arena {lens[0]:.3f} m (0.20 bars), "
        f"dragstrip {lens[1]:.3f} m (0.25 bar, 0.32 strokes)")
    # the grid is painted where the race lines cars up (the game's
    # race_grid: you at s 0 n 0, bots in rows of two at n +-2.2, rows 7 m
    # back), standing in T7's R 130 arc as the cars do: every slot's box
    # has its bar just ahead of that car's nose, and no paint under the car
    gsl = la.grid_slots
    gs_ = la.grid_s if la.grid_s is not None else np.zeros(0)
    want = [(0.0, GRID_SIDE_N), (0.0, 0.0), (0.0, -GRID_SIDE_N)] + [
        ((-GRID_ROW_M * k) % la.L, sg * GRID_SIDE_N) for k in range(1, GRID_ROWS)
        for sg in (1.0, -1.0)]
    ok_slots = sorted((round(float(a), 6), float(b)) for a, b in gsl) == sorted(
        (round(a, 6), b) for a, b in want)
    worst_k = 0.0
    for sg_ in gs_:
        ss_ = np.arange(sg_ - GRID_TAIL - 0.3, sg_ + GRID_NOSE + 0.3, 0.1)
        worst_k = max(worst_k, float(np.abs(ar.kappa[_i_of_s(ar, ss_)]).max()))
    P = la.paint
    Pc = P.mean(axis=1)
    bar_err, under = 0.0, 0
    for s_c, n_c in gsl:
        c_ = np.array(pt(ar, s_c, n_c))
        _x, _y, p_c, _k = trk._eval(ar, float(s_c))
        t_ = np.array([math.cos(p_c), math.sin(p_c)])
        l_ = np.array([-t_[1], t_[0]])
        want_bar = np.array(pt(ar, s_c + GRID_NOSE + 0.2, n_c))
        bar_err = max(bar_err, float(np.hypot(*(Pc - want_bar).T).min()))
        # every paint quad in this car's own frame: none may overlap its
        # 3.82 x 1.65 m footprint
        X, Y = (P - c_) @ t_, (P - c_) @ l_
        hit = ((X.max(axis=1) > -GRID_TAIL) & (X.min(axis=1) < GRID_NOSE)
               & (Y.max(axis=1) > -0.823) & (Y.min(axis=1) < 0.823))
        under += int(hit.sum())
    rep("grid: where the race lines cars up, in T7's arc",
        ok_slots and worst_k < 1.0 / GRID_R_MIN and bar_err < 0.05 and under == 0,
        f"{len(gsl)} boxes in {len(gs_)} rows at s "
        f"{np.round(np.where(gs_ > la.L / 2, gs_ - la.L, gs_), 1).tolist()}"
        f" (race_grid's slots: {ok_slots}); max |kappa| {worst_k:.5f} (R "
        f"{1.0 / max(worst_k, 1e-9):.0f} m > {GRID_R_MIN:.0f}); bars within "
        f"{bar_err * 100:.1f} cm of nose + 0.2 m; {under} paint quads under a car")
    # timing: four wheels, on and off the road (the per-frame call)
    P4 = np.array([pt(ar, 330.0, 5.4), pt(ar, 330.0, 4.0), pt(ar, 328.0, 5.4), pt(ar, 328.0, 4.0)])
    Pg = np.array([pt(ar, 640.0, 9.0 + d) for d in (0.0, 1.4, -0.5, 0.9)])
    for _ in range(20):
        wheel_surfaces(ar, P4, [True] * 4, [1.0] * 4)
    t0 = time.perf_counter()
    nrep = 400
    for _ in range(nrep):
        wheel_surfaces(ar, P4, [True] * 4, [1.0] * 4)
        wheel_surfaces(ar, Pg, [False] * 4, [0.55] * 4)
    us = (time.perf_counter() - t0) / (2 * nrep) * 1e6
    rep("wheel_surfaces cost (4 wheels) < 100 us", us < 100.0,
        f"{us:.1f} us per call, measured over {2 * nrep} calls")
    tb = {nm: round(lay.build_ms, 1) for nm, lay in lays.items()}
    rep("layout build (once per track)", max(tb.values()) < 2000.0, f"ms {tb}")
    rep("per-map themes", [lays[k].theme for k in trk.TRACK_ORDER]
        == ["circuit", "open", "skidpad", "dragstrip"] and len(lays["skidpad"].guide_rings) >= 3
        and len(lays["dragstrip"].paint) > 20 and lays["open"].dashes,
        f"skidpad {len(lays['skidpad'].guide_rings)} guide rings, dragstrip "
        f"{len(lays['dragstrip'].paint)} paint quads, open keeps its dashes")
    if verbose:
        print(f"  {'PASS' if ok_all else 'FAIL'}: {sum(1 for r in res if r[1])}/{len(res)} checks")
    return ok_all


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)
