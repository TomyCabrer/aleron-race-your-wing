"""Track geometry, arclength projection and per-wheel surface lookup.

Why this module exists in this shape
------------------------------------
The physics (`drive/vehicle.py`) is written entirely in the ISO body frame and
knows nothing about a road. Everything that turns a world position into
"where am I on the lap, and what is under this tyre" lives here, so the
vehicle model never imports the track and the two can be validated separately.

Three design decisions carry all the weight:

  * The centreline is defined by an EXACT segment list (straights and constant
    radius arcs) and sampled at DS = 0.5 m for drawing and for the projection
    seed -- but every number that has to be accurate (`project`, `point_at`)
    is evaluated against the exact analytic segment, never against the
    polyline. The chordal sag of a 0.5 m chord on the tightest corner (R = 30)
    is DS^2/(8R) = 1.04 mm, which is bigger than the 1e-3 m round-trip
    tolerance the harness spec demands, so a purely polyline projection
    cannot pass V3. The polyline gives the seed; two Newton steps on
    d/ds dot(p - C(s), T(s)) = 0 give the answer.

  * The nearest-sample search is an 8.0 m grid hash with an EXPLICIT
    3x3 -> 5x5 -> full-argmin fallback. An empty bucket query is the normal
    case far off track (a spin into the grass); without the fallback the
    argmin is taken over an empty candidate set and either raises or, worse,
    silently reuses a stale index and `n` flips sign.

  * Index arithmetic on a closed track is MODULAR. The chord refinement looks
    at i-1 and i+1; at i = 0 and i = N-1 a non-modular version pins `s` to
    either 0 or `length`, so the car's arclength glitches by a full lap every
    time it crosses the start line and the lap timer double-fires.

Geometry provenance
-------------------
CIRCUIT_ARENA's 14-segment list, its origin, its 14 node coordinates and its
7 arc centres come from `specs/harness.txt` verbatim: the two "odd" straight
lengths (169.2669025 and 105.2229498 m) were solved from a 2x2 linear closure
system with every other quantity fixed at a round value. They are NOT to be
re-derived -- rounding either one to 3 dp opens the loop by ~0.5 m. The three
newer circuits (Linden, Kestrel, Ashdown) were closed the same way, by
`solve_closure`, and are not in the harness spec: their literals, and the
rules their shapes answer to, are under "three more circuits" below.

Sign convention (contract section 0): +y is LEFT, psi and turn_deg positive is
counter-clockwise = a LEFT turn, and `n` from `project()` is positive to the
LEFT of the centreline. The same convention as the vehicle model, on purpose.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field, replace

import numpy as np

# --- module constants -------------------------------------------------------
DS = 0.5                     # m   centreline sample spacing (2499 pts on the arena)
GRID_CELL = 8.0              # m   projection hash cell; a 3x3 span is 24 m >= half-width
MU_OFF_TRACK = 0.55          # est, band 0.4-0.7   grass/gravel; a deterrent, not calibrated
CRR_OFF_SCALE = 25.0         # est                 Crr 0.012 -> 0.30 off track
MU_WET_SCALE = 0.632183908   # published  qss.sweep: 0.55/0.87
MU_DAMP_SCALE = 0.80         # est, band 0.75-0.88  between dry 1.0 and standing water


# --- primitives -------------------------------------------------------------
@dataclass(frozen=True)
class Seg:
    """One centreline primitive: 'S' straight or 'A' constant-radius arc.

    For an arc the caller supplies radius and turn_deg; `length` is recomputed
    here so the two can never drift apart. turn_deg > 0 is a LEFT / CCW turn.
    """

    kind: str
    length: float = 0.0
    radius: float = 0.0
    turn_deg: float = 0.0

    def __post_init__(self):
        if self.kind == "A":
            object.__setattr__(
                self, "length", abs(math.radians(self.turn_deg)) * self.radius
            )
        elif self.kind != "S":
            raise ValueError(f"Seg.kind must be 'S' or 'A', got {self.kind!r}")

    @classmethod
    def straight(cls, length: float) -> "Seg":
        return cls("S", length)

    @classmethod
    def arc(cls, radius: float, turn_deg: float) -> "Seg":
        return cls("A", 0.0, radius, turn_deg)


@dataclass(frozen=True)
class SurfacePatch:
    """A rectangle in (s, n). Overlapping patches MULTIPLY their scales.

    s0 > s1 means the window wraps the start line on a closed track.
    """

    s0: float
    s1: float
    n0: float
    n1: float
    mu_scale: float = 1.0
    crr_scale: float = 1.0
    label: str = ""
    colour: tuple = (43, 58, 74)


@dataclass(frozen=True)
class Area:
    """A drivable (or surface-scaling) region in WORLD coordinates.

    The ribbon is the only tarmac on a circuit; an open map is mostly tarmac
    with a ribbon painted on it, so it needs regions the s/n patch cannot
    express. Three analytic shapes -- every one is an O(1) containment test,
    which matters at 4 wheels x 200 Hz:

        'rrect'   (cx, cy, hx, hy, r)   rounded rectangle, corner radius r
        'rect'    (x0, y0, x1, y1)
        'circle'  (cx, cy, r)

    `drivable` areas count as tarmac (on_track); every area that contains the
    point multiplies its mu/crr scale in, exactly like a SurfacePatch.
    """

    kind: str
    params: tuple
    mu_scale: float = 1.0
    crr_scale: float = 1.0
    drivable: bool = True
    label: str = ""
    colour: tuple = (58, 61, 67)

    def contains(self, x: float, y: float) -> bool:
        p = self.params
        if self.kind == "rect":
            return p[0] <= x <= p[2] and p[1] <= y <= p[3]
        if self.kind == "circle":
            dx, dy = x - p[0], y - p[1]
            return dx * dx + dy * dy <= p[2] * p[2]
        cx, cy, hx, hy, r = p                       # rrect
        ax, ay = abs(x - cx), abs(y - cy)
        if ax > hx or ay > hy:
            return False
        qx, qy = ax - (hx - r), ay - (hy - r)
        if qx <= 0.0 or qy <= 0.0:
            return True
        return qx * qx + qy * qy <= r * r

    def polygon(self, n_arc: int = 10) -> np.ndarray:
        """(M,2) outline, counter-clockwise, for the renderer."""
        p = self.params
        if self.kind == "rect":
            return np.array([(p[0], p[1]), (p[2], p[1]), (p[2], p[3]), (p[0], p[3])])
        if self.kind == "circle":
            t = np.linspace(0.0, 2.0 * math.pi, 4 * n_arc, endpoint=False)
            return np.column_stack([p[0] + p[2] * np.cos(t), p[1] + p[2] * np.sin(t)])
        cx, cy, hx, hy, r = p
        pts = []
        corners = ((cx + hx - r, cy - hy + r, -0.5 * math.pi),
                   (cx + hx - r, cy + hy - r, 0.0),
                   (cx - hx + r, cy + hy - r, 0.5 * math.pi),
                   (cx - hx + r, cy - hy + r, math.pi))
        for (ccx, ccy, a0) in corners:
            for k in range(n_arc + 1):
                a = a0 + 0.5 * math.pi * k / n_arc
                pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
        return np.array(pts)

    def bbox(self) -> tuple:
        p = self.params
        if self.kind == "rect":
            return (p[0], p[1], p[2], p[3])
        if self.kind == "circle":
            return (p[0] - p[2], p[1] - p[2], p[0] + p[2], p[1] + p[2])
        cx, cy, hx, hy, _ = p
        return (cx - hx, cy - hy, cx + hx, cy + hy)


@dataclass
class Track:
    """Immutable description plus the arrays `build()` fills in.

    xy / left / right are (N,2) float64; s / psi / kappa are (N,). On a closed
    track the last sample is the closure point, i.e. xy[-1] == xy[0] to within
    the accumulated integration error, so the renderer can draw the ribbon as
    one polygon without a special case at the seam.
    """

    name: str
    width: float
    segs: list
    origin: tuple
    heading0: float
    closed: bool
    surfaces: list = field(default_factory=list)
    sector_s: list = field(default_factory=list)
    guide_radii: list = field(default_factory=list)   # skidpad only
    gates: list = field(default_factory=list)         # dragstrip only: (s, label)
    areas: list = field(default_factory=list)         # open map: world-space tarmac
    features: list = field(default_factory=list)      # open map: painted decorations
    #   ('circle', cx, cy, r, colour, width_m) | ('cone', x, y)
    #   ('line', x0, y0, x1, y1, colour, width_m) | ('box', x0, y0, x1, y1, colour)
    title: str = ""                                   # human name for menus / HUD

    # filled by build()
    s: np.ndarray = field(default=None, repr=False)
    xy: np.ndarray = field(default=None, repr=False)
    psi: np.ndarray = field(default=None, repr=False)
    kappa: np.ndarray = field(default=None, repr=False)
    left: np.ndarray = field(default=None, repr=False)
    right: np.ndarray = field(default=None, repr=False)
    length: float = 0.0
    bbox: tuple = ()
    ds: float = DS                      # actual spacing; length/(N-1), ~DS
    nodes: np.ndarray = field(default=None, repr=False)   # (nseg+1, 2) segment starts
    node_s: np.ndarray = field(default=None, repr=False)
    node_psi: np.ndarray = field(default=None, repr=False)
    centre: tuple = ()                  # skidpad only, for the guide circles

    # private: exact-segment tables and the projection hash
    _t0: list = field(default=None, repr=False)   # segment start arclengths
    _tk: list = field(default=None, repr=False)   # 0 straight, +1/-1 arc sign
    _tR: list = field(default=None, repr=False)
    _tx: list = field(default=None, repr=False)   # straight: p0; arc: centre
    _ty: list = field(default=None, repr=False)
    _tp: list = field(default=None, repr=False)   # psi0
    _grid: dict = field(default=None, repr=False)
    _nwrap: int = 0
    # python-list mirrors of s/xy: the chord refinement runs on scalars, and
    # numpy scalar arithmetic is ~5x slower than float and leaks np.float64
    # (and hence np.bool_) into every downstream caller of surface_at().
    _sl: list = field(default=None, repr=False)
    _xl: list = field(default=None, repr=False)
    _yl: list = field(default=None, repr=False)


# --- exact centreline evaluation -------------------------------------------
def _wrap_pi(a: float) -> float:
    """Fold an angle into (-pi, pi].

    _eval() accumulates heading through the segment list, so on the arena psi
    reaches 2*pi by T7 and on a skidpad it reaches 2*pi within one lap. Any
    caller forming (psi_car - psi_c) on the raw value picks up a spurious full
    turn exactly at the start line, so every psi this module HANDS OUT is
    wrapped; only the internal segment table stays continuous.
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _eval(tr: Track, s: float):
    """Exact (x, y, psi, kappa) at arclength s. Wraps if closed, clamps if not.

    This is the analytic curve, not the polyline. Everything that needs better
    than millimetre accuracy goes through here.
    """
    L = tr.length
    if tr.closed:
        s = s - L * math.floor(s / L)
    elif s < 0.0:
        s = 0.0
    elif s > L:
        s = L

    t0 = tr._t0
    k = bisect_right(t0, s) - 1
    if k < 0:
        k = 0
    elif k >= len(t0):
        k = len(t0) - 1
    t = s - t0[k]

    sgn = tr._tk[k]
    if sgn == 0.0:                       # straight
        p = tr._tp[k]
        return (tr._tx[k] + t * math.cos(p), tr._ty[k] + t * math.sin(p), p, 0.0)

    R = tr._tR[k]
    th = tr._tp[k] + sgn * t / R
    return (tr._tx[k] + sgn * R * math.sin(th),
            tr._ty[k] - sgn * R * math.cos(th),
            th, sgn / R)


def _integrate_nodes(tr: Track):
    """Walk the segment list once, exactly, recording the pose at each start."""
    x, y = float(tr.origin[0]), float(tr.origin[1])
    psi = float(tr.heading0)
    s = 0.0
    t0, tk, tR, tx, ty, tp = [], [], [], [], [], []
    nodes = [(x, y)]
    node_s = [0.0]
    node_psi = [psi]

    for seg in tr.segs:
        t0.append(s)
        tp.append(psi)
        if seg.kind == "S":
            tk.append(0.0)
            tR.append(0.0)
            tx.append(x)
            ty.append(y)
            x = x + seg.length * math.cos(psi)
            y = y + seg.length * math.sin(psi)
        else:
            sgn = 1.0 if seg.turn_deg >= 0.0 else -1.0
            R = seg.radius
            # centre lies to the LEFT for a left turn: c = p0 + sgn*R*[-sin, cos]
            cx = x + sgn * R * (-math.sin(psi))
            cy = y + sgn * R * (math.cos(psi))
            tk.append(sgn)
            tR.append(R)
            tx.append(cx)
            ty.append(cy)
            th = psi + sgn * seg.length / R
            x = cx + sgn * R * math.sin(th)
            y = cy - sgn * R * math.cos(th)
            psi = th
        s += seg.length
        nodes.append((x, y))
        node_s.append(s)
        node_psi.append(psi)

    tr._t0, tr._tk, tr._tR, tr._tx, tr._ty, tr._tp = t0, tk, tR, tx, ty, tp
    tr.length = s
    tr.nodes = np.asarray(nodes, dtype=float)
    tr.node_s = np.asarray(node_s, dtype=float)
    tr.node_psi = np.asarray(node_psi, dtype=float)


def build(tr: Track) -> Track:
    """Integrate the segments, sample at DS, build the edges and the hash.

    Sampling uses N-1 = round(length/DS) intervals rather than a fixed 0.5 m
    step, so the LAST sample lands exactly on s = length. On a closed track
    that is the closure point and xy[-1] == xy[0] to the integration error --
    which is what V1 measures. A fixed 0.5 m step would leave a 0.2 m stub and
    the closure assertion would be meaningless.
    """
    _integrate_nodes(tr)
    L = tr.length
    if L <= 0.0:
        raise ValueError("track has zero length")

    if tr.closed:
        turn = sum(sg.turn_deg for sg in tr.segs)
        # abs() so a clockwise skidpad (-360) passes the same assertion
        assert abs(abs(turn) - 360.0) < 1e-6, f"total heading {turn} != +/-360 deg"

    n_int = int(round(L / DS))
    n_int = max(n_int, 2)
    N = n_int + 1
    ds = L / n_int
    tr.ds = ds

    s_arr = np.empty(N)
    xy = np.empty((N, 2))
    ps = np.empty(N)
    kp = np.empty(N)
    for i in range(N - 1):
        si = ds * i
        s_arr[i] = si
        x, y, p, k = _eval(tr, si)
        xy[i, 0] = x
        xy[i, 1] = y
        ps[i] = p
        kp[i] = k
    # The last sample must be the ACCUMULATED end pose, not _eval(L): _eval
    # wraps a closed track's s modulo length and would hand back xy[0] exactly,
    # making the closure assertion below vacuous.
    s_arr[N - 1] = L
    xy[N - 1] = tr.nodes[-1]
    ps[N - 1] = tr.node_psi[-1]
    kp[N - 1] = kp[N - 2]

    if tr.closed:
        gap = float(np.hypot(xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1]))
        assert gap < 1e-4, f"centreline does not close: {gap:.3e} m"

    nl = np.column_stack((-np.sin(ps), np.cos(ps)))     # left unit normal
    h = 0.5 * tr.width
    ps = (ps + np.pi) % (2.0 * np.pi) - np.pi           # publish wrapped headings
    tr.s, tr.xy, tr.psi, tr.kappa = s_arr, xy, ps, kp
    tr.left = xy + h * nl
    tr.right = xy - h * nl
    tr.bbox = (float(xy[:, 0].min()), float(xy[:, 1].min()),
               float(xy[:, 0].max()), float(xy[:, 1].max()))

    # --- projection hash ----------------------------------------------------
    # For each cell we PRE-MERGE the 3x3 neighbourhood candidate list, so a
    # query is one dict lookup plus one vectorised argmin instead of nine dict
    # lookups and a python-level merge. Cells whose 3x3 is empty but whose 5x5
    # is not get the 5x5 list under the same key; anything else falls through
    # to the full argmin. That IS the 3x3 -> 5x5 -> full ladder, precomputed.
    tr._nwrap = N - 1 if tr.closed else N     # drop the duplicated closure point
    occ = {}
    inv = 1.0 / GRID_CELL
    for i in range(tr._nwrap):
        key = (int(math.floor(xy[i, 0] * inv)), int(math.floor(xy[i, 1] * inv)))
        b = occ.get(key)
        if b is None:
            occ[key] = [i]
        else:
            b.append(i)

    grid = {}
    for rad in (1, 2):
        for (gx, gy) in occ:
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        continue
                    cand = []
                    for ex in range(-rad, rad + 1):
                        for ey in range(-rad, rad + 1):
                            b = occ.get((key[0] + ex, key[1] + ey))
                            if b:
                                cand.extend(b)
                    if cand:
                        idx = np.asarray(cand, dtype=np.intp)
                        grid[key] = (idx, xy[idx, 0].copy(), xy[idx, 1].copy())
    tr._grid = grid
    tr._sl = s_arr.tolist()
    tr._xl = xy[:, 0].tolist()
    tr._yl = xy[:, 1].tolist()
    return tr


# --- projection -------------------------------------------------------------
def project(tr: Track, x: float, y: float):
    """(s, n, kappa_track, psi_c, i). n > 0 is to the LEFT of the centreline.

    Grid-hashed nearest sample -> chord refinement across the two adjacent
    chords (modular on a closed track) -> two Newton steps against the exact
    segment. The Newton step is
        s <- s + dot(p - C, T) / (1 - kappa*n)
    which is the derivative of the projection residual along the curve; it is
    what buys the 1e-3 m round-trip accuracy the chord alone cannot reach.
    """
    inv = 1.0 / GRID_CELL
    key = (int(math.floor(x * inv)), int(math.floor(y * inv)))
    cell = tr._grid.get(key)
    if cell is not None:
        cx, cy = cell[1], cell[2]
        dx = cx - x
        dy = cy - y
        i = int(cell[0][int(np.argmin(dx * dx + dy * dy))])
    else:
        # full argmin: only reached well off track (>16 m from the ribbon)
        pxy = tr.xy
        dx = pxy[:tr._nwrap, 0] - x
        dy = pxy[:tr._nwrap, 1] - y
        i = int(np.argmin(dx * dx + dy * dy))

    xl = tr._xl
    yl = tr._yl
    s_arr = tr._sl
    nw = tr._nwrap
    ds = tr.ds
    xi = xl[i]
    yi = yl[i]

    # chord refinement on i-1..i and i..i+1, modular when closed
    best_d = float("inf")
    s_est = s_arr[i]
    for step in (-1, 1):
        j = i + step
        if tr.closed:
            j %= nw
        elif j < 0 or j >= nw:
            continue
        ex = xl[j] - xi
        ey = yl[j] - yi
        den = ex * ex + ey * ey
        if den <= 0.0:
            continue
        u = ((x - xi) * ex + (y - yi) * ey) / den
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        qx = xi + u * ex - x
        qy = yi + u * ey - y
        d = qx * qx + qy * qy
        if d < best_d:
            best_d = d
            s_est = s_arr[i] + step * u * ds

    for _ in range(2):
        cx_, cy_, pc, kc = _eval(tr, s_est)
        dx = x - cx_
        dy = y - cy_
        cs = math.cos(pc)
        sn = math.sin(pc)
        e = dx * cs + dy * sn
        n = -dx * sn + dy * cs
        den = 1.0 - kc * n
        if abs(den) < 1e-6:
            break
        s_est += e / den

    cx_, cy_, pc, kc = _eval(tr, s_est)
    dx = x - cx_
    dy = y - cy_
    n = -dx * math.sin(pc) + dy * math.cos(pc)

    L = tr.length
    if tr.closed:
        s_out = s_est - L * math.floor(s_est / L)
    else:
        s_out = 0.0 if s_est < 0.0 else (L if s_est > L else s_est)

    idx = int(round(s_out / ds))
    if tr.closed:
        idx %= nw
    elif idx >= nw:
        idx = nw - 1
    return (s_out, n, kc, _wrap_pi(pc), idx)


def point_at(tr: Track, s: float, n: float = 0.0):
    """Inverse of project(): world (x, y) at arclength s, offset n to the LEFT."""
    x, y, p, _ = _eval(tr, s)
    return (x - n * math.sin(p), y + n * math.cos(p))


def start_pose(tr: Track, offset_n: float = 0.0):
    """(x, y, psi) on the grid: s = 0, displaced offset_n to the LEFT."""
    x, y, p, _ = _eval(tr, 0.0)
    return (x - offset_n * math.sin(p), y + offset_n * math.cos(p), _wrap_pi(p))


# --- surfaces ---------------------------------------------------------------
def _patch_hit(pt: SurfacePatch, s: float, n: float) -> bool:
    if not (pt.n0 <= n <= pt.n1):
        return False
    if pt.s1 >= pt.s0:
        return pt.s0 <= s <= pt.s1
    return s >= pt.s0 or s <= pt.s1        # window wraps the start line


def surface_at(tr: Track, x: float, y: float, global_wet: float = 1.0):
    """(mu_scale, crr_scale, on_track) at one world point -- ONE WHEEL.

    Called four times per wheel-update (200 Hz) by the harness. A single
    car-centre lookup would silently delete split-mu, which is the whole
    reason DAMP_T5_EXIT and the dragstrip split patch exist.
    """
    s, n, _, _, _ = project(tr, x, y)
    on = abs(n) <= 0.5 * tr.width
    if on and not tr.closed:
        # On an OPEN track project() clamps s to [0, length], so a point 80 m
        # off the end of the dragstrip still reports a small |n| and would read
        # as tarmac. Reconstructing the point from (s, n) catches the
        # longitudinal overshoot; a closed track can never have one.
        px, py = point_at(tr, s, n)
        if (px - x) ** 2 + (py - y) ** 2 > 1e-8:
            on = False
    mu = 1.0
    crr = 1.0
    if tr.areas:
        # an open map: the ribbon OR any drivable area is tarmac, and every
        # area that contains the point scales the surface (a wet square)
        for ar in tr.areas:
            if ar.contains(x, y):
                if ar.drivable:
                    on = True
                mu *= ar.mu_scale
                crr *= ar.crr_scale
    if not on:
        mu *= MU_OFF_TRACK
        crr *= CRR_OFF_SCALE
    for pt in tr.surfaces:
        if _patch_hit(pt, s, n):
            mu *= pt.mu_scale
            crr *= pt.crr_scale
    return (mu * global_wet, crr, on)


def on_tarmac(tr: Track, x: float, y: float, n: float | None = None) -> bool:
    """Is the point on the ribbon or on a drivable area? `n` may be passed
    by a caller that has already projected the point (drive.Sim does, every
    step) so the closed-track case costs no second project()."""
    if n is None:
        n = project(tr, x, y)[1]
    if abs(n) <= 0.5 * tr.width:
        if tr.closed:
            return True
        s = project(tr, x, y)[0]
        px, py = point_at(tr, s, n)
        if (px - x) ** 2 + (py - y) ** 2 <= 1e-8:
            return True
    for ar in tr.areas:
        if ar.drivable and ar.contains(x, y):
            return True
    return False


# --- literal geometry -------------------------------------------------------
# specs/harness.txt, verbatim. The two odd straights were solved from a 2x2
# closure system with everything else round; do not re-derive or round them.
CIRCUIT_ARENA_SEGS = [
    Seg.straight(169.2669025),
    Seg.arc(45.0, 100.0),      # T1
    Seg.straight(80.0),
    Seg.arc(30.0, 120.0),      # T2  tightest
    Seg.straight(60.0),
    Seg.arc(80.0, -100.0),     # T3  the only right-hander pair with T7
    Seg.straight(50.0),
    Seg.arc(55.0, 110.0),      # T4
    Seg.straight(105.2229498),
    Seg.arc(110.0, 80.0),      # T5
    Seg.straight(40.0),
    Seg.arc(35.0, 100.0),      # T6
    Seg.straight(30.0),
    Seg.arc(130.0, -50.0),     # T7  the only POWER-limited corner (qss: 31.88 m/s)
]
CIRCUIT_ARENA_ORIGIN = (213.20402134, 84.41894405)

# Chosen to put standing water in the fastest grip-limited corner and split-mu
# in a braking zone. mu values are published (0.632 = qss.sweep 0.55/0.87);
# the 0.80 damp scale is an estimate, band 0.75-0.88.
ARENA_PATCHES = [
    SurfacePatch(455.0, 585.0, -6.0, 6.0, mu_scale=MU_WET_SCALE,
                 label="WET_T3", colour=(43, 58, 74)),
    SurfacePatch(960.0, 1010.0, -6.0, 0.0, mu_scale=MU_DAMP_SCALE,
                 label="DAMP_T5_EXIT", colour=(51, 57, 63)),
    SurfacePatch(300.0, 330.0, -6.0, 0.0, mu_scale=MU_WET_SCALE,
                 label="WET_T2_ENTRY", colour=(43, 58, 74)),
]

# Optional (spec: "optional"); not installed by make_dragstrip() so the V11-V13
# straight-line acceptance numbers are measured on clean tarmac. drive.py adds
# them for --wet patch.
DRAGSTRIP_PATCHES = [
    SurfacePatch(1050.0, 1120.0, -7.5, 7.5, mu_scale=MU_WET_SCALE,
                 label="DRAG_WET", colour=(43, 58, 74)),
    SurfacePatch(1050.0, 1150.0, -7.5, 0.0, mu_scale=MU_WET_SCALE,
                 label="DRAG_SPLIT", colour=(43, 58, 74)),
]
DRAGSTRIP_BOARDS = [900.0, 1000.0, 1100.0, 1200.0]   # brake boards, m


def solve_closure(segs, ia: int, ib: int, heading0: float = 0.0):
    """(La, Lb): the lengths of straights `ia` and `ib` that close the loop.

    How every circuit's two "odd" straights were found. The end point of a
    segment walk is LINEAR in the length of any straight (the heading does
    not depend on it), so with every other segment fixed at a round value
    the gap to close is

        La * [cos th_a, sin th_a] + Lb * [cos th_b, sin th_b] = -(x, y)

    where (x, y) is the displacement of all the OTHER segments and th_a,
    th_b the headings at the two free straights: a 2x2 system with
    det = sin(th_b - th_a). The lengths `segs[ia]` / `segs[ib]` carry are
    ignored. Pick the two legs roughly at right angles (det near 1): nearly
    parallel legs make the system ill-conditioned, and a NEGATIVE answer
    means a leg points the wrong way for this shape.

    Only the self-check calls this: the factories store the answers as
    literals at 7 dp (closure ~5e-8 m), like the arena's, so building a
    track never solves anything and a stored number cannot silently drift.
    """
    x = y = 0.0
    psi = float(heading0)
    dirs = {}
    for k, sg in enumerate(segs):
        if k in (ia, ib):
            if sg.kind != "S":
                raise ValueError(f"segment {k} is not a straight")
            dirs[k] = (math.cos(psi), math.sin(psi))
            continue
        if sg.kind == "S":
            x += sg.length * math.cos(psi)
            y += sg.length * math.sin(psi)
        else:
            sgn = 1.0 if sg.turn_deg >= 0.0 else -1.0
            th = psi + math.radians(sg.turn_deg)
            x += sgn * sg.radius * (math.sin(th) - math.sin(psi))
            y -= sgn * sg.radius * (math.cos(th) - math.cos(psi))
            psi = th
    (a1, a2), (b1, b2) = dirs[ia], dirs[ib]
    det = a1 * b2 - a2 * b1
    return ((-x * b2 + y * b1) / det, (-a1 * y + a2 * x) / det)


def make_arena(surfaces: bool = True) -> Track:
    """The 1249.2022 m closed circuit. Seven corners spanning R = 30..130 m.

    `surfaces` is a keyword with a default so `make_arena()` matches the API in
    the contract while drive.py can still honour `--wet none` by passing False.
    The spec calls the patches "off by default", but V26/V27 drive through them,
    so they are installed unless explicitly suppressed.
    """
    tr = Track(
        name="arena",
        width=12.0,                      # two-lane racing width
        segs=list(CIRCUIT_ARENA_SEGS),
        origin=CIRCUIT_ARENA_ORIGIN,
        heading0=0.0,
        closed=True,
        surfaces=[replace(p) for p in ARENA_PATCHES] if surfaces else [],
        sector_s=[0.0, 390.639, 851.080],
    )
    return build(tr)


# --- three more circuits ----------------------------------------------------
# Built the arena's way: straights and constant-radius arcs, every quantity
# round except two straights solved by `solve_closure` and stored at 7 dp
# (CLOSURE_FREE says which two). Nothing else about them is special-cased
# anywhere: the dressing (kerbs, gravel, grid boxes, pits, stands, barriers)
# is derived from the geometry by scenery.py and props.py, so each shape was
# chosen to satisfy the rules those modules place things by --
#   * width 12 m and R 30..130 m, the band the scripted driver, the ML anchor
#     and the projection's 8 m hash were all calibrated on;
#   * s = 0 part-way along a straight with >= 37.4 m of it behind the line
#     (six painted grid rows need R >= 80 there) and >= 120 m through it
#     (the pits need Lp >= 36 m of a 0.3-0.62 share of it);
#   * centrelines >= 60 m apart wherever they are > 90 m apart in s, i.e. the
#     width plus two scenery.DRESS_MAX run-off bands, so the gravel of one
#     stretch never meets the next and the 30 m-clear props have room;
#   * the sector lines on straights (the reset and a race respawn stand the
#     car on them), sector_s[0] == 0.0 (the lap line, the grid, the gantry);
#   * every surface patch covers n = 0: drive.ml's observation reads the
#     grip on the centreline only, and a patch it cannot see is one the bots
#     and the anchor arrive at at dry speed. As on the arena, one full-width
#     wet patch sits in a fast grip-limited corner and one half-width wet
#     strip in a braking zone, on the corner's turn-in side (Ashdown's
#     racing line crosses its strip rather than braking in it: see there);
#   * the two reference drivers (medals.py drives both) lap it in all three
#     cars. That ruled out three shapes the rules above allow, all measured:
#     a straight of 300 m or more into a corner wider than R ~70 (the ML
#     anchor discounts its 120 m lookahead, K120_PLAN, so for an R90 bend
#     it plans 47 m/s there and only starts braking at the 70 m station --
#     the 540i and the MX-5 went off); a fast corner, a 60 m straight and a
#     heavy stop (LapDriver brakes below its target by turn-in and the speed
#     loop floors the throttle mid-corner: the Corsa washes wide, the MX-5
#     spins); and an R130 kink taken at the grip limit less than ~150 m
#     before a heavy stop. At an arc's end LapDriver's throttle cap (the
#     friction ellipse on the PATH's curvature) steps to 1 and the gearbox
#     kicks down, so the MX-5 leaves a limit R130 swaying; aids off, the
#     full-pedal stop then locks its fronts while it drifts, the steering
#     winds up, and the release throws it off at turn-in. The arena's R130
#     (T7) never reaches its grip limit: it is POWER-limited, 30 m after the
#     R35 T6. Kestrel's and Ashdown's docstrings say where.
# The self-check re-solves the closure and re-measures all of the above.
CIRCUIT_LINDEN_SEGS = [                # tight and technical, 150 m pit straight
    Seg.straight(100.0),
    Seg.arc(35.0, 90.0),       # T1
    Seg.straight(284.4920075),
    Seg.arc(30.0, 150.0),      # T2  hairpin, tightest
    Seg.straight(60.0),
    Seg.arc(50.0, -60.0),      # T3  right
    Seg.straight(109.7560269),
    Seg.arc(40.0, 90.0),       # T4
    Seg.straight(50.0),
    Seg.arc(60.0, 45.0),       # T5  the fastest grip-limited corner: wet
    Seg.straight(40.0),
    Seg.arc(45.0, -45.0),      # T6  right
    Seg.straight(30.0),
    Seg.arc(35.0, 90.0),       # T7
    Seg.straight(50.0),
]
CIRCUIT_LINDEN_ORIGIN = (159.0380592, 15.0)
LINDEN_PATCHES = [
    SurfacePatch(850.0, 905.0, -6.0, 6.0, mu_scale=MU_WET_SCALE,
                 label="WET_T5", colour=(43, 58, 74)),
    SurfacePatch(400.0, 435.0, -6.0, 0.0, mu_scale=MU_WET_SCALE,
                 label="WET_T2_ENTRY", colour=(43, 58, 74)),
]

CIRCUIT_KESTREL_SEGS = [               # fast: 390 m pit and 320 m back straights, R 55..100
    Seg.straight(320.0),
    Seg.arc(60.0, 60.0),       # T1  the stop at the end of the pit straight
    Seg.straight(80.0),
    Seg.arc(80.0, 60.0),       # T2  fast and grip-limited: wet
    Seg.straight(120.0),
    Seg.arc(100.0, -40.0),     # T3  right, the fast kink (not R130: make_kestrel)
    Seg.straight(140.0),
    Seg.arc(55.0, 120.0),      # T4  tightest
    Seg.straight(320.0),
    Seg.arc(70.0, 70.0),       # T5
    Seg.straight(172.7266405),
    Seg.arc(60.0, -50.0),      # T6  right
    Seg.straight(50.0),
    Seg.arc(70.0, 140.0),      # T7  the long one onto the pit straight
    Seg.straight(70.0815616),
]
CIRCUIT_KESTREL_ORIGIN = (155.08154234, 15.0)
KESTREL_PATCHES = [
    SurfacePatch(466.0, 544.0, -6.0, 6.0, mu_scale=MU_WET_SCALE,
                 label="WET_T2", colour=(43, 58, 74)),
    SurfacePatch(830.0, 870.0, -6.0, 0.0, mu_scale=MU_WET_SCALE,
                 label="WET_T4_ENTRY", colour=(43, 58, 74)),
]

CIRCUIT_ASHDOWN_SEGS = [               # CLOCKWISE (sum -360): mostly right-handers
    Seg.straight(151.2626648),
    Seg.arc(40.0, -90.0),      # T1
    Seg.straight(60.0),
    Seg.arc(100.0, -45.0),     # T2
    Seg.straight(80.0),
    Seg.arc(30.0, 100.0),      # T3  the only left, tightest
    Seg.straight(50.0),
    Seg.arc(35.0, -150.0),     # T4  hairpin
    Seg.straight(200.0),
    Seg.arc(120.0, -30.0),     # T5  the fastest grip-limited corner: wet
    Seg.straight(90.0),
    Seg.arc(55.0, -85.0),      # T6
    Seg.straight(190.4464162),
    Seg.arc(75.0, -60.0),      # T7
    Seg.straight(60.0),
]
CIRCUIT_ASHDOWN_ORIGIN = (242.54325522, 380.41608798)
ASHDOWN_PATCHES = [
    #  T5, not T2: in T2 the ML anchor, seeing the water 60 m ahead,
    #  lifted out of T1 and spun the 540i
    SurfacePatch(830.0, 886.0, -6.0, 6.0, mu_scale=MU_WET_SCALE,
                 label="WET_T5", colour=(43, 58, 74)),
    #  n 0..6, the LEFT half, T4's turn-in side (T4 turns right). T4 comes
    #  only 50 m after the left-hand T3, too close to set up from the
    #  outside, so the racing line runs apex to apex and CROSSES the strip:
    #  n +0.8 -> -1.8 over 495..530, leaving it at s ~506. What the strip
    #  is for holds all the same: a car on the centreline (LapDriver, the
    #  anchor, a bot) brakes with its left wheels in the water, split-mu
    SurfacePatch(495.0, 530.0, 0.0, 6.0, mu_scale=MU_WET_SCALE,
                 label="WET_T4_ENTRY", colour=(43, 58, 74)),
]

#: the two straights `solve_closure` solved on each circuit (seg indices)
CLOSURE_FREE = {"arena": (0, 8), "linden": (2, 6), "kestrel": (10, 14),
                "ashdown": (0, 12)}


def _circuit(name, segs, origin, patches, sector_s, title, surfaces) -> Track:
    """One of the three circuits below: the arena's width, closed, heading 0."""
    tr = Track(
        name=name,
        width=12.0,
        segs=list(segs),
        origin=origin,
        heading0=0.0,
        closed=True,
        surfaces=[replace(p) for p in patches] if surfaces else [],
        sector_s=list(sector_s),
        title=title,
    )
    return build(tr)


def make_linden(surfaces: bool = True) -> Track:
    """Linden park: 1110.4021 m, counter-clockwise, tight and technical.

    Seven corners at R 30..60 m and a 284 m back straight into the R30
    hairpin; the shortest lap of the circuits. `surfaces` as make_arena."""
    return _circuit("linden", CIRCUIT_LINDEN_SEGS, CIRCUIT_LINDEN_ORIGIN,
                    LINDEN_PATCHES, [0.0, 370.0, 690.0], "Linden park", surfaces)


def make_kestrel(surfaces: bool = True) -> Track:
    """Kestrel ring: 1913.3440 m, counter-clockwise, fast.

    Seven corners at R 55..100 m, a 390 m pit straight (across the line)
    and a 320 m back straight; the longest lap.
    Each long straight ends in a stop (T1 R60, T5 R70) rather than a fast
    bend: the first draft ran 357 / 376 m straights into R110 / R90 and the
    ML anchor put the MX-5 off at T1 and the 540i off at T5 (see the
    section comment).

    T3 -> T4 was the other reshape. The spec's R130 T3, 60 m from an R45
    T4, spun LapDriver's MX-5 in T4; at 100 m the MX-5 still left the road
    there with aids off on the DRY surface (the wet strip only hid it: the
    plan braked earlier and softer for it). A longer straight into an R55
    fixed that and broke another class instead: the tuned MX-5's
    full-throttle exit slide grew into a spin before the brakes came on.
    The cause is the R130 itself (the section comment), so T3 is R100,
    140 m from an R55 T4. Measured with medals.py's own runs:
    LapDriver 0.90 laps every class the 100 m draft did plus four more
    (among them the MX-5's stock dry class, aids off), the anchor laps both
    surfaces in all three cars, and the neighbours (T3 R90..110, 120..160
    m, T4 R45..60) keep the MX-5's lap. `surfaces` as make_arena."""
    return _circuit("kestrel", CIRCUIT_KESTREL_SEGS, CIRCUIT_KESTREL_ORIGIN,
                    KESTREL_PATCHES, [0.0, 610.0, 1140.0], "Kestrel ring", surfaces)


def make_ashdown(surfaces: bool = True) -> Track:
    """Ashdown circuit: 1390.0362 m, CLOCKWISE, mixed.

    Six right-handers and one left at R 30..120 m, including the R35
    hairpin; the only circuit whose infield is on the right (props.py reads
    the side from the sign of the total turn). T5 -> T6 is 90 m: at 60 m
    LapDriver left the road at T6 in every car (measured |n| 14.5 m in the
    Corsa). `surfaces` as make_arena."""
    return _circuit("ashdown", CIRCUIT_ASHDOWN_SEGS, CIRCUIT_ASHDOWN_ORIGIN,
                    ASHDOWN_PATCHES, [0.0, 505.0, 930.0], "Ashdown circuit", surfaces)


def make_skidpad(radius: float = 50.0, cw: bool = False) -> Track:
    """A single guide circle about (200, 200) as the centreline.

    Making the driven circle the centreline (rather than painting circles on a
    disc) means s/n projection, lap timing and the surface lookup all work
    unchanged. The other guide radii are carried in guide_radii for the
    renderer. R = 50 and R = 100 are qss.py's stated validation radii,
    R = 130 is the power-limit crossover, R = 30.48 is the ISO 200 ft pad.
    """
    cx, cy = 200.0, 200.0
    sgn = -1.0 if cw else 1.0
    tr = Track(
        name="skidpad",
        width=20.0,
        segs=[Seg.arc(radius, sgn * 360.0)],
        origin=(cx + radius, cy),
        heading0=sgn * math.pi / 2.0,
        closed=True,
        surfaces=[],
        sector_s=[0.0],
        guide_radii=[30.48, 50.0, 75.0, 100.0, 130.0],
    )
    tr = build(tr)
    tr.centre = (cx, cy)
    return tr


def make_dragstrip() -> Track:
    """1500 m straight. Long enough that the car asymptotes toward Vmax 47.2."""
    tr = Track(
        name="dragstrip",
        width=15.0,
        segs=[Seg.straight(1500.0)],
        origin=(20.0, 40.0),
        heading0=0.0,
        closed=False,
        surfaces=[],
        sector_s=[0.0],
        gates=[(0.0, "launch"), (201.168, "1/8 mile"),
               (402.336, "1/4 mile"), (1000.0, "standing km")],
    )
    return build(tr)


# --- the open map: a proving ground ----------------------------------------
# One rounded rectangle of tarmac, 522 x 362 m, with a 12 m perimeter road
# painted round its edge as the CENTRELINE (so s/n projection, lap and sector
# timing, the sector-line reset and the minimap all work unchanged) and, inside
# it, the things a chassis engineer walks a car through: a skidpad (ISO 200 ft
# circle and the R=50 m circle qss.py validates on), a 9-cone slalom at 18 m,
# a 300 m marked drag lane, and a wet square (mu 0.632, the published
# qss.sweep wet figure). Off the pad is grass. Nothing here is a spec number:
# the geometry is chosen so the whole pad fits ~2.5 screens at the wide zoom.
OPEN_STRAIGHT_X = 420.0      # m   bottom / top straights
OPEN_STRAIGHT_Y = 260.0      # m   left / right straights
OPEN_R = 45.0                # m   perimeter corner radius (kerbs: |kappa| > 1/60)
OPEN_WIDTH = 12.0            # m   perimeter road
OPEN_MARGIN = 6.0            # m   tarmac past the road's outer edge


def make_open() -> Track:
    """The proving ground. Closed perimeter loop, everything inside is tarmac."""
    R, W = OPEN_R, OPEN_WIDTH
    sx, sy = OPEN_STRAIGHT_X, OPEN_STRAIGHT_Y
    segs = [Seg.straight(sx), Seg.arc(R, 90.0),
            Seg.straight(sy), Seg.arc(R, 90.0),
            Seg.straight(sx), Seg.arc(R, 90.0),
            Seg.straight(sy), Seg.arc(R, 90.0)]
    L = 2.0 * (sx + sy) + 2.0 * math.pi * R
    # the pad: centreline rectangle [0, sx] x [0, sy] grown by R + W/2 + margin
    edge = R + 0.5 * W + OPEN_MARGIN
    cx, cy = 0.5 * sx, 0.5 * sy
    hx, hy = 0.5 * sx + edge, 0.5 * sy + edge
    pad = Area("rrect", (cx, cy, hx, hy, edge), label="PAD",
               colour=(52, 55, 60))
    wet = Area("rect", (250.0, 95.0, 370.0, 175.0), mu_scale=MU_WET_SCALE,
               drivable=True, label="WET_SQUARE", colour=(43, 58, 74))
    # painted features (drawn only; the physics never reads them)
    C_PAINT = (150, 154, 160)
    feats = [
        ("circle", 105.0, 150.0, 50.0, C_PAINT, 0.15),        # qss R = 50 m
        ("circle", 105.0, 150.0, 30.48, C_PAINT, 0.15),       # ISO 200 ft pad
        ("circle", 105.0, 150.0, 1.0, C_PAINT, 0.4),          # centre dot
        ("line", 60.0, 40.0, 360.0, 40.0, C_PAINT, 0.12),     # drag lane, 300 m
        ("line", 60.0, 46.0, 360.0, 46.0, C_PAINT, 0.12),
        ("box", 60.0, 40.0, 62.0, 46.0, (216, 218, 222)),      # launch box
    ]
    for k in range(1, 4):                                     # 100 m boards
        xk = 60.0 + 100.0 * k
        feats.append(("line", xk, 38.5, xk, 47.5, (216, 218, 222), 0.25))
    for k in range(9):                                        # slalom, 18 m
        feats.append(("cone", 200.0 + 18.0 * k, 232.0))
    for k in range(6):                                        # a gate pair
        feats.append(("cone", 30.0 + 8.0 * k, 232.0 + 3.0))
        feats.append(("cone", 30.0 + 8.0 * k, 232.0 - 3.0))
    tr = Track(
        name="open",
        width=W,
        segs=segs,
        origin=(0.0, 0.0),
        heading0=0.0,
        closed=True,
        surfaces=[],
        sector_s=[0.0, round(L / 3.0, 3), round(2.0 * L / 3.0, 3)],
        areas=[pad, wet],
        features=feats,
        title="Open proving ground",
    )
    return build(tr)


TRACKS: dict = {
    "arena": make_arena,
    "linden": make_linden,
    "kestrel": make_kestrel,
    "ashdown": make_ashdown,
    "open": make_open,
    "skidpad": make_skidpad,
    "dragstrip": make_dragstrip,
}
TRACK_TITLES: dict = {
    "arena": "Arena circuit",
    "linden": "Linden park",
    "kestrel": "Kestrel ring",
    "ashdown": "Ashdown circuit",
    "open": "Open proving ground",
    "skidpad": "Skidpad",
    "dragstrip": "Dragstrip",
}
#: TAB cycles this: the circuits together, then the test maps. A name is a
#: plain lowercase identifier with no '_', ',' or '|' and never
#: '<another map>_...': it is a records class-key field and file-name part,
#: a checkpoint's comma-listed meta and the seed-lap glob `seed_<map>_`.
TRACK_ORDER = ("arena", "linden", "kestrel", "ashdown", "open", "skidpad", "dragstrip")
#: The race CIRCUITS: closed, 12 m, dressed with the circuit theme, lapped
#: with records and medals. `Track.closed` does not say this (the open map's
#: perimeter and the skidpad close too).
CIRCUITS = ("arena", "linden", "kestrel", "ashdown")


def make_track(name: str, radius: float = 50.0, cw: bool = False,
               surfaces: bool = True) -> Track:
    """One place that turns a name + options into a built Track.

    `radius` / `cw` are the skidpad's, `surfaces` every circuit's (False is
    `--wet none`). An unknown name builds the ARENA: drive/ml/evaluate.py
    hands a multi-track checkpoint meta ('arena,open') straight in and relies
    on that. The flip side is that a map missing a branch here silently
    builds the arena under its own name, so self_check asserts
    `make_track(n).name == n` for every name in TRACK_ORDER."""
    if name == "skidpad":
        return make_skidpad(radius, cw)
    if name == "dragstrip":
        return make_dragstrip()
    if name == "open":
        return make_open()
    if name == "linden":
        return make_linden(surfaces=surfaces)
    if name == "kestrel":
        return make_kestrel(surfaces=surfaces)
    if name == "ashdown":
        return make_ashdown(surfaces=surfaces)
    return make_arena(surfaces=surfaces)


# --- self-check -------------------------------------------------------------
_ARENA_NODES = [                     # specs/harness.txt, pasted as a fixture
    (0.0, 213.204, 84.419), (169.267, 382.471, 84.419),
    (247.807, 426.787, 137.233), (327.807, 412.895, 216.018),
    (390.639, 364.068, 233.790), (450.639, 318.105, 195.222),
    (590.265, 197.400, 216.506), (640.265, 172.400, 259.807),
    (745.857, 82.636, 267.661), (851.080, 15.000, 187.055),
    (1004.669, 15.000, 45.642), (1044.669, 40.712, 15.000),
    (1105.756, 94.335, 15.000), (1135.756, 113.618, 37.981),
]
_ARENA_CENTRES = [(382.471, 129.419), (383.351, 210.808), (266.682, 256.506),
                  (124.768, 232.307), (99.265, 116.348), (67.523, 37.498),
                  (213.204, -45.581)]

# the three circuits' lengths, pasted as fixtures (a changed shape must be
# a deliberate edit here too: it makes the medal table stale)
_CIRCUIT_LENGTHS = {"linden": 1110.4021, "kestrel": 1913.3440, "ashdown": 1390.0362}
_CIRCUIT_CW = {"linden": False, "kestrel": False, "ashdown": True}
#: centrelines this far apart wherever they are > 90 m apart in s: the width
#: plus two scenery.DRESS_MAX (24 m) run-off bands, so no two stretches'
#: dressing meets (the arena, drawn before the rule, has 65.55)
CLEARANCE_MIN = 60.0


def _min_separation(tr: Track, gap_s: float = 90.0) -> float:
    """Closest approach of the centreline to itself, over sample pairs more
    than `gap_s` apart in arclength (V4). 90 m is about pi * R_min: any two
    points of one R30 hairpin are nearer than that along the road."""
    P = tr.xy[:-1]
    sv = tr.s[:-1]
    L = tr.length
    dmin = float("inf")
    for i in range(0, len(P)):
        dsep = np.abs(sv - sv[i])
        dsep = np.minimum(dsep, L - dsep)
        m = dsep > gap_s
        if not m.any():
            continue
        d = np.hypot(P[m, 0] - P[i, 0], P[m, 1] - P[i, 1]).min()
        if d < dmin:
            dmin = float(d)
    return dmin


def _circuit_checks(rep, verbose: bool) -> None:
    """V1-V4 for the three newer circuits, plus the rules their comment
    lists (grid zone, sector lines, radii, patches on the centreline)."""
    segs_of = {"arena": CIRCUIT_ARENA_SEGS, "linden": CIRCUIT_LINDEN_SEGS,
               "kestrel": CIRCUIT_KESTREL_SEGS, "ashdown": CIRCUIT_ASHDOWN_SEGS}
    worst = 0.0
    for name, (ia, ib) in CLOSURE_FREE.items():
        segs = segs_of[name]
        La, Lb = solve_closure(segs, ia, ib)
        worst = max(worst, abs(La - segs[ia].length), abs(Lb - segs[ib].length))
    rep("closure re-solved", worst < 1e-6,
        f"solve_closure reproduces every circuit's two stored straights to "
        f"{worst:.1e} m ({', '.join(CLOSURE_FREE)})")
    for name in ("linden", "kestrel", "ashdown"):
        if verbose:
            print(f"circuit  {name} ({TRACK_TITLES[name]})")
        tr = make_track(name)
        hw = 0.5 * tr.width
        gap = float(np.hypot(*(tr.xy[-1] - tr.xy[0])))
        turn = sum(sg.turn_deg for sg in tr.segs)
        rep(f"{name} closure", gap < 1e-4 and abs(abs(turn) - 360.0) < 1e-6
            and (turn < 0.0) == _CIRCUIT_CW[name] and tr.closed and tr.width == 12.0,
            f"|xy[-1]-xy[0]| = {gap:.1e} m, sum(turn_deg) = {turn:+.1f} "
            f"({'clockwise' if turn < 0 else 'counter-clockwise'}), width {tr.width:.0f}")
        rep(f"{name} length", abs(tr.length - _CIRCUIT_LENGTHS[name]) < 0.01,
            f"length = {tr.length:.4f} m  (expect {_CIRCUIT_LENGTHS[name]:.4f})")
        rng = np.random.default_rng(12345)
        ss = rng.uniform(0.0, tr.length, 500)
        nn = rng.uniform(-hw, hw, 500)
        dsm = dnm = 0.0
        for a, b in zip(ss, nn):
            px, py = point_at(tr, float(a), float(b))
            s2, n2, _, _, _ = project(tr, px, py)
            e = abs(s2 - a)
            dsm = max(dsm, min(e, tr.length - e))
            dnm = max(dnm, abs(n2 - b))
        rep(f"{name} round trip", dsm < 1e-3 and dnm < 1e-3,
            f"500 random (s, |n| <= {hw:.0f}): max |ds| {dsm:.1e} m, max |dn| {dnm:.1e} m")
        dmin = _min_separation(tr)
        rep(f"{name} clearance", dmin >= CLEARANCE_MIN,
            f"min separation = {dmin:.2f} m (>= {CLEARANCE_MIN:.0f}: width + 2 x 24 m run-off)")
        x0, y0, p0 = start_pose(tr)
        radii = [sg.radius for sg in tr.segs if sg.kind == "A"]
        rep(f"{name} start and radii",
            abs(x0 - tr.origin[0]) < 1e-9 and abs(y0 - tr.origin[1]) < 1e-9
            and abs(p0 - tr.heading0) < 1e-12 and 30.0 <= min(radii) and max(radii) <= 130.0,
            f"start ({x0:.3f}, {y0:.3f}) heading {math.degrees(p0):.1f} deg; "
            f"R {min(radii):.0f}..{max(radii):.0f} m (band 30..130)")
        sec = list(tr.sector_s)
        on_straight = all(abs(_eval(tr, v + d)[3]) == 0.0 for v in sec for d in (-2.0, 0.0, 2.0))
        rep(f"{name} sector lines", len(sec) == 3 and sec[0] == 0.0 and sec == sorted(sec)
            and sec[-1] < tr.length and on_straight,
            f"{sec}, each on a straight (+-2 m)")
        kg = max(abs(_eval(tr, float(v))[3]) for v in np.arange(-37.4, 2.1, 0.1))
        rep(f"{name} grid zone", kg <= 1.0 / 80.0,
            f"max |kappa| over s -37.4..+2.1 = {kg:.4f} (<= 1/80: six painted rows)")
        seen = []
        for pt in tr.surfaces:
            idx = np.nonzero((tr.s >= pt.s0) & (tr.s <= pt.s1))[0]
            mus = [surface_at(tr, float(tr.xy[i, 0]), float(tr.xy[i, 1]))[0] for i in idx]
            seen.append(pt.n0 <= 0.0 <= pt.n1 and len(idx) > 0
                        and all(abs(m - pt.mu_scale) < 1e-12 for m in mus))
        dry = make_track(name, surfaces=False)
        rep(f"{name} patches on the centreline",
            len(tr.surfaces) == 2 and all(seen) and dry.surfaces == []
            and tr.title == TRACK_TITLES[name],
            ", ".join(f"{p.label} {p.s0:.0f}..{p.s1:.0f} n {p.n0:+.0f}..{p.n1:+.0f}"
                      for p in tr.surfaces) + "; every centreline sample inside reads its mu; "
            "surfaces=False drops them")
    rep("CIRCUITS", all(n in TRACK_ORDER and n in TRACKS for n in CIRCUITS)
        and all(make_track(n).closed for n in CIRCUITS)
        and tuple(n for n in TRACK_ORDER if n in CIRCUITS) == CIRCUITS,
        f"{', '.join(CIRCUITS)}: closed, in TRACK_ORDER and TRACKS, in menu order")


# Corsa C wheel contact points in the body frame (contract / harness geometry):
# front axle x = +0.9715, rear x = -1.5195; half tracks 0.7145 and 0.7100.
_WHEEL_XY = ((0.9715, 0.7145), (0.9715, -0.7145),
             (-1.5195, 0.7100), (-1.5195, -0.7100))


def _wheel_points(tr, s_car, n_car=0.0):
    x, y, p, _ = _eval(tr, s_car)
    x -= n_car * math.sin(p)
    y += n_car * math.cos(p)
    c, sn = math.cos(p), math.sin(p)
    return [(x + bx * c - by * sn, y + bx * sn + by * c) for bx, by in _WHEEL_XY]


def self_check(verbose: bool = True) -> bool:
    import time

    ok = True

    def rep(tag, passed, msg):
        nonlocal ok
        ok = ok and passed
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {tag}: {msg}")

    tr = make_arena()
    if verbose:
        print("V1  CIRCUIT_ARENA closure / length / bbox")
    gap = float(np.hypot(*(tr.xy[-1] - tr.xy[0])))
    turn = sum(sg.turn_deg for sg in tr.segs)
    rep("closure", gap < 1e-4, f"|xy[-1]-xy[0]| = {gap:.4e} m  (expect < 1e-4, spec 3.8e-8)")
    rep("heading", abs(turn - 360.0) < 1e-6, f"sum(turn_deg) = {turn:.6f} deg")
    rep("length", abs(tr.length - 1249.2022) < 0.01,
        f"length = {tr.length:.4f} m  (expect 1249.2022)")
    nb = (float(tr.nodes[:, 0].min()), float(tr.nodes[:, 1].min()),
          float(tr.nodes[:, 0].max()), float(tr.nodes[:, 1].max()))
    rep("bbox(nodes)", max(abs(a - b) for a, b in
                           zip(nb, (15.0, 15.0, 426.787, 267.661))) < 0.005,
        "node bbox (%.3f, %.3f) .. (%.3f, %.3f)" % nb)
    if verbose:
        print("       centreline bbox (%.3f, %.3f) .. (%.3f, %.3f)"
              "   <- arcs bulge outside the node bbox the spec quotes" % tr.bbox)
        print(f"       samples N = {len(tr.s)}, ds = {tr.ds:.6f} m")

    if verbose:
        print("V2  14 node coordinates")
    worst = 0.0
    for k, (sk, xk, yk) in enumerate(_ARENA_NODES):
        d = math.hypot(tr.nodes[k, 0] - xk, tr.nodes[k, 1] - yk)
        worst = max(worst, d)
        if abs(tr.node_s[k] - sk) > 0.002:
            worst = 1e9
    rep("nodes", worst < 0.002, f"max node error = {worst:.6f} m over 14 nodes")
    wc = 0.0
    ai = [i for i, sg in enumerate(tr.segs) if sg.kind == "A"]
    for k, (cxx, cyy) in zip(ai, _ARENA_CENTRES):
        wc = max(wc, math.hypot(tr._tx[k] - cxx, tr._ty[k] - cyy))
    rep("arc centres", wc < 0.002, f"max arc-centre error = {wc:.6f} m over 7 arcs")

    if verbose:
        print("V3  projection round trip, 500 random (s, n)")
    rng = np.random.default_rng(12345)
    ss = rng.uniform(0.0, tr.length, 500)
    nn = rng.uniform(-6.0, 6.0, 500)
    dsm = dnm = 0.0
    pts = [point_at(tr, float(a), float(b)) for a, b in zip(ss, nn)]
    for (px, py), a, b in zip(pts, ss, nn):
        s2, n2, _, _, _ = project(tr, px, py)
        e = abs(s2 - a)
        e = min(e, tr.length - e)
        dsm = max(dsm, e)
        dnm = max(dnm, abs(n2 - b))
    t0 = time.perf_counter()
    for (px, py) in pts:
        project(tr, px, py)
    us = (time.perf_counter() - t0) / len(pts) * 1e6
    rep("round trip", dsm < 1e-3 and dnm < 1e-3,
        f"max |ds| = {dsm:.3e} m, max |dn| = {dnm:.3e} m")
    rep("speed", us < 20.0, f"mean project() = {us:.2f} us/call  (budget 20)")

    # far off track: the 3x3 and 5x5 buckets are both empty here
    s_far, n_far, _, _, i_far = project(tr, -200.0, -200.0)
    d_far = math.hypot(*[a - b for a, b in
                         zip(point_at(tr, s_far, n_far), (-200.0, -200.0))])
    rep("off-track fallback", d_far < 1e-6 and abs(n_far) > 100.0,
        f"p(-200,-200) -> s {s_far:.2f}, n {n_far:.2f}, reconstructed to {d_far:.2e} m")

    if verbose:
        print("V4  self-clearance (points > 90 m apart in arclength)")
    dmin = _min_separation(tr)
    rep("clearance", dmin > 20.0,
        f"min separation = {dmin:.2f} m  (spec 65.96; width is 12)")

    if verbose:
        print("V26 WET_T3 boundary, per wheel, driving the centreline")
    step = 0.01
    errs = []
    for edge, before, after in ((455.0, 1.0, MU_WET_SCALE),
                                (585.0, MU_WET_SCALE, 1.0)):
        for w in range(4):
            lo, hi = edge - 6.0, edge + 6.0
            prev = None
            flip = None
            sc = lo
            while sc <= hi:
                mu = surface_at(tr, *_wheel_points(tr, sc)[w])[0]
                if prev is not None and abs(mu - prev) > 1e-9:
                    flip = sc
                    break
                prev = mu
                sc += step
            if flip is None:
                errs.append(99.9)
            else:
                # the wheel is offset longitudinally from the car origin, so
                # convert the car s at which it flipped into the wheel's own s
                wx, wy = _wheel_points(tr, flip)[w]
                errs.append(abs(project(tr, wx, wy)[0] - edge))
    e26 = max(errs)
    rep("wet boundary", e26 < 0.20,
        f"max |s_flip - s_edge| = {e26:.4f} m over 8 (wheel, edge) cases")
    mid = surface_at(tr, *_wheel_points(tr, 520.0)[0])[0]
    rep("wet value", abs(mid - MU_WET_SCALE) < 1e-9,
        f"mu inside WET_T3 = {mid:.6f} (expect {MU_WET_SCALE:.6f})")

    if verbose:
        print("V27 split-mu in DAMP_T5_EXIT (n -6..0 = the RIGHT half)")
    mus = [surface_at(tr, wx, wy)[0] for wx, wy in _wheel_points(tr, 985.0)]
    rep("split mu", abs(mus[0] - 1.0) < 1e-12 and abs(mus[2] - 1.0) < 1e-12
        and abs(mus[1] - 0.80) < 1e-12 and abs(mus[3] - 0.80) < 1e-12,
        "FL %.3f  FR %.3f  RL %.3f  RR %.3f" % tuple(mus))

    if verbose:
        print("extra  skidpad / dragstrip")
    sp = make_skidpad(50.0)
    x0, y0, p0 = start_pose(sp)
    rep("skidpad pose", abs(x0 - 250.0) < 1e-9 and abs(y0 - 200.0) < 1e-9
        and abs(p0 - math.pi / 2) < 1e-12,
        f"start ({x0:.3f}, {y0:.3f}) psi {math.degrees(p0):.3f} deg, "
        f"length {sp.length:.4f} (2*pi*50 = {2 * math.pi * 50:.4f})")
    cxs = [math.hypot(px - 200.0, py - 200.0) for px, py in sp.xy]
    rep("skidpad radius", max(abs(c - 50.0) for c in cxs) < 1e-8,
        f"max |r - 50| = {max(abs(c - 50.0) for c in cxs):.2e} m")
    spc = make_skidpad(50.0, cw=True)
    rep("skidpad cw", abs(start_pose(spc)[2] + math.pi / 2) < 1e-12,
        f"cw start psi = {math.degrees(start_pose(spc)[2]):.3f} deg")
    dg = make_dragstrip()
    s_d, n_d, _, _, _ = project(dg, 20.0 + 402.336, 40.0 + 3.0)
    rep("dragstrip", abs(s_d - 402.336) < 1e-9 and abs(n_d - 3.0) < 1e-9
        and dg.gates[2] == (402.336, "1/4 mile"),
        f"1/4 mile gate: s {s_d:.4f}, n {n_d:.4f}, length {dg.length:.1f}")
    off_end = surface_at(dg, 1600.0, 40.0)
    off_side = surface_at(dg, 700.0, 40.0 + 9.0)
    rep("open-track ends", off_end[2] is False and off_side[2] is False
        and abs(off_end[0] - MU_OFF_TRACK) < 1e-12,
        f"100 m past the end -> mu {off_end[0]:.3f} on_track {off_end[2]}; "
        f"9 m to the side -> on_track {off_side[2]}")

    if verbose:
        print("extra  open map (proving ground)")
    op = make_open()
    L_exp = 2.0 * (OPEN_STRAIGHT_X + OPEN_STRAIGHT_Y) + 2.0 * math.pi * OPEN_R
    gap = float(np.hypot(*(op.xy[-1] - op.xy[0])))
    rep("open closure", gap < 1e-6 and abs(op.length - L_exp) < 1e-6,
        f"closure {gap:.2e} m, length {op.length:.3f} m (expect {L_exp:.3f})")
    x0, y0, p0 = start_pose(op)
    rep("open start pose", abs(x0) < 1e-9 and abs(y0) < 1e-9 and abs(p0) < 1e-12,
        f"({x0:.3f}, {y0:.3f}) heading {math.degrees(p0):.1f} deg")
    mid = surface_at(op, 105.0, 150.0)            # skidpad centre, far from the road
    wetc = surface_at(op, 300.0, 130.0)           # inside the wet square
    road = surface_at(op, 210.0, 0.0)             # on the bottom straight
    off = surface_at(op, 210.0, -80.0)            # grass, south of the pad
    corner = surface_at(op, -50.0, -50.0)         # outside the rounded corner
    rep("open pad is tarmac", mid[2] is True and abs(mid[0] - 1.0) < 1e-12
        and road[2] is True and abs(road[0] - 1.0) < 1e-12,
        f"pad centre mu {mid[0]:.3f} on {mid[2]}; road mu {road[0]:.3f} on {road[2]}")
    rep("open wet square", wetc[2] is True and abs(wetc[0] - MU_WET_SCALE) < 1e-12,
        f"mu {wetc[0]:.3f} on {wetc[2]}")
    rep("open off-pad is grass", off[2] is False and abs(off[0] - MU_OFF_TRACK) < 1e-12
        and corner[2] is False and abs(corner[1] - CRR_OFF_SCALE) < 1e-12,
        f"south mu {off[0]:.3f} on {off[2]}; corner cut-off on {corner[2]} crr {corner[1]:.0f}")
    rep("on_tarmac agrees with surface_at",
        on_tarmac(op, 105.0, 150.0) and not on_tarmac(op, 210.0, -80.0)
        and on_tarmac(tr, *point_at(tr, 100.0, 3.0)) and not on_tarmac(tr, -200.0, -200.0),
        "pad centre / grass / arena ribbon / arena far off")
    t0 = time.perf_counter()
    for k in range(2000):
        surface_at(op, 20.0 + 0.2 * k, 10.0 + 0.1 * k)
    us = (time.perf_counter() - t0) / 2000 * 1e6
    rep("open surface_at speed", us < 30.0, f"{us:.2f} us/call (budget 30)")
    poly = op.areas[0].polygon()
    rep("open pad polygon", len(poly) == 44 and abs(poly[:, 0].min() - (-OPEN_R - 0.5 * OPEN_WIDTH - OPEN_MARGIN)) < 1e-9,
        f"{len(poly)} vertices, x {poly[:, 0].min():.1f}..{poly[:, 0].max():.1f}, "
        f"y {poly[:, 1].min():.1f}..{poly[:, 1].max():.1f}")

    if verbose:
        print("extra  the circuits (Linden, Kestrel, Ashdown)")
    _circuit_checks(rep, verbose)
    rep("make_track by name", all(make_track(n).name == n for n in TRACK_ORDER),
        ", ".join(TRACK_ORDER))

    if verbose:
        print(f"\n{'ALL TRACK CHECKS PASS' if ok else 'TRACK CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if self_check() else 1)
